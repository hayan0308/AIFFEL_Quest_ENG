import os
# konlpy가 Java를 필요로 하므로 JAVA_HOME을 먼저 설정
os.environ.setdefault(
    "JAVA_HOME",
    r"C:\Users\harao\AppData\Local\Programs\Eclipse Adoptium\jdk-25.0.2.10-hotspot"
)

import re
import random
import math
import urllib.request

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from konlpy.tag import Okt  # 윈도우에서는 Mecab 대신 Okt 사용
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 40

# =====================================================================
# Step 1. 데이터 다운로드
# =====================================================================
csv_path = "ChatbotData.csv"
if not os.path.exists(csv_path):
    url = "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv"
    urllib.request.urlretrieve(url, csv_path)

df = pd.read_csv(csv_path)
questions = df["Q"].tolist()
answers = df["A"].tolist()
print("데이터 개수:", len(questions))


# =====================================================================
# Step 2. 데이터 정제
# =====================================================================
def preprocess_sentence(sentence):
    sentence = sentence.lower()
    # 한글, 영문자, 숫자, 주요 특수문자(.,!?)만 남기고 제거
    sentence = re.sub(r"[^0-9a-zA-Z가-힣.,!?]+", " ", sentence)
    sentence = re.sub(r" {2,}", " ", sentence)
    sentence = sentence.strip()
    return sentence


# =====================================================================
# Step 3. 데이터 토큰화
# =====================================================================
okt = Okt()

def build_corpus(src_sentences, tgt_sentences, tokenize_fn, max_len=MAX_LEN):
    src_seen, tgt_seen = set(), set()
    src_corpus, tgt_corpus = [], []

    for src, tgt in tqdm(list(zip(src_sentences, tgt_sentences))):
        src_clean = preprocess_sentence(src)
        tgt_clean = preprocess_sentence(tgt)

        src_tokens = tokenize_fn(src_clean)
        tgt_tokens = tokenize_fn(tgt_clean)

        if len(src_tokens) > max_len or len(tgt_tokens) > max_len:
            continue
        if src_clean in src_seen or tgt_clean in tgt_seen:
            continue

        src_seen.add(src_clean)
        tgt_seen.add(tgt_clean)
        src_corpus.append(src_tokens)
        tgt_corpus.append(tgt_tokens)

    return src_corpus, tgt_corpus


que_corpus, ans_corpus = build_corpus(questions, answers, okt.morphs)
print("정제 후 데이터 개수:", len(que_corpus))


# =====================================================================
# Step 4. Augmentation (Lexical Substitution)
# =====================================================================
# ko.bin은 Kyubyong/wordvectors 저장소의 "Korean (w)" 링크(Google Drive)에서
# 직접 다운로드해야 합니다: https://github.com/Kyubyong/wordvectors
# 다운로드한 ko.bin 파일을 이 스크립트와 같은 폴더에 두세요.
from gensim.models import KeyedVectors
import pickle as _pickle

# ---- ko.bin: 구버전 gensim pickle → 현재 KeyedVectors 변환 ----
class _OldVocab:
    def __init__(self, **kw): self.__dict__.update(kw)

class _FakeRandomState:
    def __setstate__(self, state): pass
    def set_state(self, state): pass
    def __reduce__(self): return (_FakeRandomState, ())

import gensim.models.word2vec as _w2v_mod
import gensim.models.keyedvectors as _kv_mod
_w2v_mod.Vocab = _OldVocab
_kv_mod.Vocab = _OldVocab

class _CompatUnpickler(_pickle.Unpickler):
    def find_class(self, module, name):
        if 'RandomState' in name:
            return _FakeRandomState
        if name == 'Vocab':
            return _OldVocab
        return super().find_class(module, name)

wv = None
try:
    print("ko.bin 로딩 중 (구버전 gensim → 현재 변환)...")
    with open("ko.bin", "rb") as _f:
        _raw = _CompatUnpickler(_f, encoding='bytes').load()

    # syn0: (vocab_size, 200) 벡터 행렬, index2word: 단어 목록
    _syn0 = _raw.__dict__.get('syn0') or _raw.__dict__.get(b'syn0')
    _i2w  = _raw.__dict__.get('index2word') or _raw.__dict__.get(b'index2word')

    # bytes → str 디코딩
    _words = [w.decode('utf-8', errors='replace') if isinstance(w, bytes) else w
              for w in _i2w]

    wv = KeyedVectors(vector_size=_syn0.shape[1])
    wv.add_vectors(_words, _syn0)
    print(f"ko.bin 로드 성공! 단어 수: {len(_words)}, 벡터 차원: {_syn0.shape[1]}")
except Exception as _e:
    print(f"ko.bin 로드 실패: {_e}")
    print("Augmentation 없이 원본 데이터만 사용합니다.")


def lexical_sub_tokens(tokens, wv):
    valid_tokens = [tok for tok in tokens if tok in wv]
    if not valid_tokens:
        return tokens
    selected_tok = random.choice(valid_tokens)
    similar_word = wv.most_similar(selected_tok)[0][0]
    return [similar_word if tok == selected_tok else tok for tok in tokens]


# ko.bin 로드 성공 시 Augmentation, 실패 시 원본 데이터만 사용
if wv is not None:
    que_aug = [lexical_sub_tokens(q, wv) for q in tqdm(que_corpus)]
    ans_aug = [lexical_sub_tokens(a, wv) for a in tqdm(ans_corpus)]
    # 원본 + (증강된 질문, 원본 답변) + (원본 질문, 증강된 답변) => 약 3배로 확장
    src_corpus_all = que_corpus + que_aug + que_corpus
    tgt_corpus_all = ans_corpus + ans_corpus + ans_aug
else:
    # Augmentation 스킵 - 원본 데이터만 사용
    src_corpus_all = que_corpus
    tgt_corpus_all = ans_corpus
print("Augmentation 후 데이터 개수:", len(src_corpus_all))



# =====================================================================
# Step 5. 데이터 벡터화
# =====================================================================
tgt_corpus_all = [["<start>"] + tokens + ["<end>"] for tokens in tgt_corpus_all]

def build_vocab(corpora):
    vocab = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
    for corpus in corpora:
        for tokens in corpus:
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab([src_corpus_all, tgt_corpus_all])
idx2word = {idx: word for word, idx in vocab.items()}
VOCAB_SIZE = len(vocab)
print("Vocab Size:", VOCAB_SIZE)


def tokens_to_ids(tokens, vocab, max_len=MAX_LEN):
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [vocab["<pad>"]] * (max_len - len(ids))
    return ids


enc_train = torch.tensor([tokens_to_ids(t, vocab) for t in src_corpus_all], dtype=torch.long)
dec_train = torch.tensor([tokens_to_ids(t, vocab) for t in tgt_corpus_all], dtype=torch.long)
print(enc_train.shape, dec_train.shape)

BATCH_SIZE = 64
train_dataset = TensorDataset(enc_train, dec_train)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)


# =====================================================================
# Step 6. 훈련하기 (이전 노드에서 만든 Transformer 재사용)
# =====================================================================
def positional_encoding(pos, d_model):
    def cal_angle(position, i):
        return position / np.power(10000, (2 * (i // 2)) / np.float32(d_model))

    def get_posi_angle_vec(position):
        return [cal_angle(position, i) for i in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(pos)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return sinusoid_table


def generate_padding_mask(seq):
    return (seq == 0).unsqueeze(1).unsqueeze(2).float()


def generate_lookahead_mask(size):
    return torch.triu(torch.ones(size, size), diagonal=1)


def generate_masks(src, tgt):
    enc_mask = generate_padding_mask(src)
    dec_enc_mask = generate_padding_mask(src)

    dec_lookahead_mask = generate_lookahead_mask(tgt.shape[1]).unsqueeze(0).unsqueeze(1).to(device)
    dec_tgt_padding_mask = generate_padding_mask(tgt).to(device)
    dec_mask = torch.max(dec_tgt_padding_mask, dec_lookahead_mask)
    return enc_mask, dec_enc_mask, dec_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        QK = torch.matmul(Q, K.transpose(-1, -2))
        scaled_qk = QK / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        if mask is not None:
            scaled_qk = scaled_qk + (mask * -1e9)
        attentions = F.softmax(scaled_qk, dim=-1)
        out = torch.matmul(attentions, V)
        return out, attentions

    def split_heads(self, x):
        bsz, seq_len, _ = x.size()
        x = x.view(bsz, seq_len, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def combine_heads(self, x):
        bsz, num_heads, seq_len, depth = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(bsz, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        WQ, WK, WV = self.W_q(Q), self.W_k(K), self.W_v(V)
        WQ, WK, WV = self.split_heads(WQ), self.split_heads(WK), self.split_heads(WV)
        out, attn = self.scaled_dot_product_attention(WQ, WK, WV, mask)
        out = self.combine_heads(out)
        return self.linear(out), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)
        self.norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.do = nn.Dropout(dropout)

    def forward(self, x, mask):
        residual = x
        out = self.norm_1(x)
        out, attn = self.enc_self_attn(out, out, out, mask)
        out = residual + self.do(out)

        residual = out
        out = self.norm_2(out)
        out = self.ffn(out)
        out = residual + self.do(out)
        return out, attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, n_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)
        self.norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm_3 = nn.LayerNorm(d_model, eps=1e-6)
        self.do = nn.Dropout(dropout)

    def forward(self, x, enc_out, dec_enc_mask, padding_mask):
        residual = x
        out = self.norm_1(x)
        out, dec_attn = self.dec_self_attn(out, out, out, mask=padding_mask)
        out = residual + self.do(out)

        residual = out
        out = self.norm_2(out)
        out, dec_enc_attn = self.enc_dec_attn(out, enc_out, enc_out, mask=dec_enc_mask)
        out = residual + self.do(out)

        residual = out
        out = self.norm_3(out)
        out = self.ffn(out)
        out = residual + self.do(out)
        return out, dec_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, x, mask):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attns.append(attn)
        return x, attns


class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, x, enc_out, dec_enc_mask, padding_mask):
        dec_attns, dec_enc_attns = [], []
        for layer in self.layers:
            x, dec_attn, dec_enc_attn = layer(x, enc_out, dec_enc_mask, padding_mask)
            dec_attns.append(dec_attn)
            dec_enc_attns.append(dec_enc_attn)
        return x, dec_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff,
                 src_vocab_size, tgt_vocab_size, pos_len,
                 dropout=0.2, shared_fc=True, shared_emb=True):
        super().__init__()
        self.d_model = float(d_model)

        if shared_emb:
            self.enc_emb = self.dec_emb = nn.Embedding(src_vocab_size, d_model)
        else:
            self.enc_emb = nn.Embedding(src_vocab_size, d_model)
            self.dec_emb = nn.Embedding(tgt_vocab_size, d_model)

        pos_encoding_np = positional_encoding(pos_len, d_model)
        self.register_buffer("pos_encoding", torch.tensor(pos_encoding_np, dtype=torch.float32))

        self.do = nn.Dropout(dropout)
        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.fc = nn.Linear(d_model, tgt_vocab_size)

        self.shared_fc = shared_fc
        if shared_fc:
            self.fc.weight = self.dec_emb.weight

    def embedding(self, emb, x):
        seq_len = x.size(1)
        out = emb(x)
        if self.shared_fc:
            out = out * math.sqrt(self.d_model)
        out = out + self.pos_encoding[:seq_len, :].unsqueeze(0)
        return self.do(out)

    def forward(self, enc_in, dec_in, enc_mask, dec_enc_mask, dec_mask):
        enc_in_emb = self.embedding(self.enc_emb, enc_in)
        dec_in_emb = self.embedding(self.dec_emb, dec_in)

        enc_out, enc_attns = self.encoder(enc_in_emb, enc_mask)
        dec_out, dec_attns, dec_enc_attns = self.decoder(dec_in_emb, enc_out, dec_enc_mask, dec_mask)
        logits = self.fc(dec_out)
        return logits, enc_attns, dec_attns, dec_enc_attns


# ---- 하이퍼파라미터 (강의 예시 값을 시작점으로 사용, 필요시 직접 튜닝하세요) ----
n_layers = 1
d_model = 368
n_heads = 8
d_ff = 1024
dropout = 0.2
WARMUP_STEPS = 1000
EPOCHS = 10

transformer = Transformer(
    n_layers=n_layers,
    d_model=d_model,
    n_heads=n_heads,
    d_ff=d_ff,
    src_vocab_size=VOCAB_SIZE,
    tgt_vocab_size=VOCAB_SIZE,
    pos_len=MAX_LEN + 10,
    dropout=dropout,
    shared_fc=True,
    shared_emb=True,
).to(device)


class LearningRateScheduler:
    def __init__(self, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = float(step)
        arg1 = step ** -0.5
        arg2 = step * (self.warmup_steps ** -1.5)
        return (self.d_model ** -0.5) * min(arg1, arg2)


learning_rate = LearningRateScheduler(d_model, warmup_steps=WARMUP_STEPS)
optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate(1), betas=(0.9, 0.98), eps=1e-9)


def loss_function(real, pred):
    real = real.to(device)
    pred = pred.to(device)
    loss_ = F.cross_entropy(pred.contiguous().view(-1, pred.size(-1)), real.contiguous().view(-1), reduction='none')
    loss_ = loss_.view(real.size())
    mask = (real != 0).float()
    loss_ = loss_ * mask
    return loss_.sum() / mask.sum()


def train_step(src, tgt, model, optimizer):
    model.train()
    optimizer.zero_grad()

    tgt_in = tgt[:, :-1]
    gold = tgt[:, 1:]

    enc_mask, dec_enc_mask, dec_mask = generate_masks(src, tgt_in)
    src = src.to(device)
    tgt_in = tgt_in.to(device)
    enc_mask = enc_mask.to(device)
    dec_enc_mask = dec_enc_mask.to(device)
    dec_mask = dec_mask.to(device)

    predictions, enc_attns, dec_attns, dec_enc_attns = model(src, tgt_in, enc_mask, dec_enc_mask, dec_mask)
    loss = loss_function(gold, predictions)

    loss.backward()
    optimizer.step()
    return loss


if __name__ == "__main__":
    for epoch in range(EPOCHS):
        total_loss = 0.0
        dataset_count = len(train_dataloader)
        tqdm_bar = tqdm(total=dataset_count)
        for step, (src, tgt) in enumerate(train_dataloader, start=1):
            # 스텝마다 learning rate 갱신
            for g in optimizer.param_groups:
                g["lr"] = learning_rate(epoch * dataset_count + step)

            loss = train_step(src, tgt, transformer, optimizer)
            total_loss += loss.item()
            tqdm_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
            tqdm_bar.update(1)
        tqdm_bar.close()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / dataset_count:.4f}")


    # =====================================================================
    # Step 6 (계속). 예문에 대한 답변 생성
    # =====================================================================
    def translate(sentence, model, vocab, idx2word, tokenize_fn, max_len=MAX_LEN):
        model.eval()
        clean = preprocess_sentence(sentence)
        tokens = tokenize_fn(clean)
        ids = tokens_to_ids(tokens, vocab, max_len)
        enc_in = torch.tensor([ids], dtype=torch.long, device=device)

        dec_in = torch.tensor([[vocab["<start>"]]], dtype=torch.long, device=device)
        result_tokens = []

        for _ in range(max_len):
            enc_mask, dec_enc_mask, dec_mask = generate_masks(enc_in, dec_in)
            predictions, _, _, _ = model(enc_in, dec_in, enc_mask, dec_enc_mask, dec_mask)
            predicted_id = predictions[0, -1].softmax(dim=-1).argmax(dim=-1).item()

            if predicted_id == vocab["<end>"]:
                result_tokens.append("<end>")
                break

            result_tokens.append(idx2word.get(predicted_id, "<unk>"))
            new_token = torch.tensor([[predicted_id]], dtype=torch.long, device=device)
            dec_in = torch.cat([dec_in, new_token], dim=1)

        return " ".join(result_tokens)


    examples = [
        "지루하다, 놀러가고 싶어.",
        "오늘 일찍 일어났더니 피곤하다.",
        "간만에 여자친구랑 데이트 하기로 했어.",
        "집에 있는다는 소리야.",
    ]

    print("\n=== 예문 테스트 결과 ===")
    for ex in examples:
        print(f"Q: {ex}")
        print(f"A: {translate(ex, transformer, vocab, idx2word, okt.morphs)}\n")


    # =====================================================================
    # Step 7. 성능 측정하기 (BLEU Score)
    # =====================================================================
    def calculate_bleu(reference, candidate, weights=[0.25, 0.25, 0.25, 0.25]):
        return sentence_bleu([reference], candidate, weights=weights,
                              smoothing_function=SmoothingFunction().method1)


    # 원본 정제 데이터(que_corpus, ans_corpus) 중 일부를 테스트로 사용해 BLEU 측정
    sample_size = min(200, len(que_corpus))
    total_score = 0.0
    count = 0

    for i in tqdm(range(sample_size)):
        src_sentence = " ".join(que_corpus[i])
        ref_tokens = ans_corpus[i]  # <start>/<end> 제외한 원본 정답 토큰
        candidate = translate(src_sentence, transformer, vocab, idx2word, okt.morphs)
        candidate_tokens = [t for t in candidate.split() if t != "<end>"]

        score = calculate_bleu(ref_tokens, candidate_tokens)
        total_score += score
        count += 1

    print("\n평균 BLEU Score:", total_score / count)