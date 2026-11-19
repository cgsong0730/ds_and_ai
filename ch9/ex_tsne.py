from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

X = load_digits().data # (1797, 64)

X_pca = PCA(n_components=2).fit_transform(X)
X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)

print(X_pca.shape, X_tsne.shape)

