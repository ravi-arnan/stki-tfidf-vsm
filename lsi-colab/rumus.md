# 📐 Rumus MinHash (Penjelasan Lengkap)

## 1. Jaccard Similarity (dasar MinHash)

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

Di mana A dan B adalah dua himpunan (dokumen/set) yang dibandingkan.

## 2. Rumus Hash Function MinHash

$$
h(x) = (ax + b) \mod p
$$

- $a, b$ = bilangan acak (konstanta)
- $p$ = bilangan prima besar
- $x$ = elemen dari himpunan

## 3. MinHash Signature

$$
minhash_k(S) = \min_{x \in S} h_k(x)
$$

Yaitu nilai minimum dari fungsi hash ke-k yang diterapkan pada semua elemen himpunan S.

## 4. Estimasi Jaccard Similarity via MinHash

$$
\hat{J}(A, B) = \frac{1}{K} \sum_{k=1}^{K} 1[minhash_k(A) = minhash_k(B)]
$$

Di mana $K$ adalah jumlah fungsi hash yang digunakan. Semakin besar $K$, semakin akurat estimasinya.