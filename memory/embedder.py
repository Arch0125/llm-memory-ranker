import hashlib

from .utils import normalize_vector, tokenize


class BaseEmbedder:
    model_name = "base"
    dim = 0

    def embed(self, text):
        return self.embed_many([text])[0]

    def embed_many(self, texts):
        raise NotImplementedError


class HashingEmbedder(BaseEmbedder):
    def __init__(self, dim=384, n_features_per_token=4):
        self.dim = dim
        self.n_features_per_token = n_features_per_token
        self.model_name = f"hash-{dim}"

    def _update_from_token(self, vector, token, weight):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for offset in range(self.n_features_per_token):
            start = offset * 4
            idx = int.from_bytes(digest[start : start + 2], "big") % self.dim
            sign = 1.0 if digest[start + 2] % 2 == 0 else -1.0
            scale = 0.5 + (digest[start + 3] / 255.0)
            vector[idx] += sign * scale * weight

    def embed_many(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0] * self.dim
            tokens = tokenize(text, drop_stopwords=False)
            if not tokens:
                vectors.append(vector)
                continue
            for token in tokens:
                self._update_from_token(vector, token, 1.0)
            lowered = text.lower()
            for i in range(max(0, len(lowered) - 2)):
                gram = lowered[i : i + 3]
                if " " in gram:
                    continue
                self._update_from_token(vector, gram, 0.35)
            vectors.append(normalize_vector(vector))
        return vectors


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for non-hashing embeddings. "
                "Install it with `pip install sentence-transformers`."
            ) from exc

        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_many(self, texts):
        values = self.model.encode(texts, normalize_embeddings=True)
        return [list(map(float, value)) for value in values]


def build_embedder(model_name="hash-384"):
    if model_name in {"hash", "hash-384"}:
        return HashingEmbedder(dim=384)
    if model_name.startswith("hash-"):
        _, dim = model_name.split("-", 1)
        return HashingEmbedder(dim=int(dim))
    if model_name.startswith("sentence-transformers:"):
        return SentenceTransformerEmbedder(model_name.split(":", 1)[1])
    return SentenceTransformerEmbedder(model_name)
