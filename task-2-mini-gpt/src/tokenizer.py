"""手写 byte-level BPE tokenizer。

思路(与 GPT-2 一致,但从零实现):
  - 基础词表 = 256 个原始字节,天然覆盖任意 Unicode 文本,encode/decode 必然可逆
  - 训练 = 反复统计相邻 token 对的频次,把最高频的一对合并成新 token(id 从 256 递增)
  - 编码 = 对文本的 UTF-8 字节序列,按训练时的合并顺序(rank 越小优先级越高)贪心应用 merge
  - 中文一个字 3 字节,高频字会先被 merge 成单 token,低频字保持 2-3 个 token

接口约定(README「实现约定」):encode / decode / vocab_size / from_pretrained。
"""
import json
from collections import Counter
from pathlib import Path


class BPETokenizer:
    def __init__(self, merges=None):
        # merges: [(a, b), ...] 按训练顺序;第 i 个 merge 生成的新 token id = 256 + i
        self.merges = [tuple(m) for m in (merges or [])]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        # 每个 token id 对应的原始字节串,decode 时直接拼接
        self._id_bytes = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self._id_bytes.append(self._id_bytes[a] + self._id_bytes[b])

    # ----------------------------- 基本属性 -----------------------------

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    # ----------------------------- 编码/解码 -----------------------------

    @staticmethod
    def _merge_pair(ids, pair, new_id):
        """把 ids 里所有相邻的 pair 替换为 new_id(一次线性扫描)。"""
        out = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def _bpe(self, ids):
        """对一段字节 id 序列贪心应用 merge:每轮找当前序列中 rank 最小的可合并对。"""
        while len(ids) >= 2:
            best_rank, best_pair = None, None
            for pair in zip(ids, ids[1:]):
                r = self.ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, pair
            if best_pair is None:
                break
            ids = self._merge_pair(ids, best_pair, 256 + best_rank)
        return ids

    def encode(self, text):
        """text -> List[int]。按行分块跑 BPE(merge 不会跨行,训练时同理),控制单块开销。"""
        ids = []
        for chunk in text.splitlines(keepends=True):
            ids.extend(self._bpe(list(chunk.encode("utf-8"))))
        return ids

    def decode(self, ids):
        """List[int] -> text。拼回字节串再整体 UTF-8 解码,天然处理跨 token 的多字节字符。"""
        data = b"".join(self._id_bytes[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    # ----------------------------- 训练 -----------------------------

    @classmethod
    def train(cls, text, vocab_size=1024, max_bytes=400_000, verbose=False):
        """在语料上训练 merge 表。

        纯 Python 逐对统计是 O(样本长 × merge 数),所以对语料采样 max_bytes;
        采样对 merge 质量影响很小(高频对在前几十万字节里已充分体现)。
        换行符不参与 merge(作为行边界),保证 encode 按行分块与训练一致。
        """
        sample = text[: max_bytes]  # 字符数上界,编码后字节数只会更多,再截一次
        lines = [list(line.encode("utf-8")) for line in sample.splitlines()]
        total = 0
        for i, ln in enumerate(lines):
            total += len(ln)
            if total > max_bytes:
                lines = lines[: i + 1]
                break

        merges = []
        n_merges = vocab_size - 256
        for step in range(n_merges):
            counts = Counter()
            for ln in lines:
                counts.update(zip(ln, ln[1:]))
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]
            if freq < 2:
                break
            new_id = 256 + len(merges)
            merges.append(pair)
            lines = [cls._merge_pair(ln, pair, new_id) for ln in lines]
            if verbose and (step + 1) % 100 == 0:
                print(f"  merge {step + 1}/{n_merges}: {pair} -> {new_id} (freq={freq})")
        return cls(merges)

    # ----------------------------- 存取 -----------------------------

    def save(self, path):
        Path(path).write_text(json.dumps({
            "type": "byte-level-bpe",
            "vocab_size": self.vocab_size,
            "merges": [list(m) for m in self.merges],
        }, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path):
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(merges=obj["merges"])
