"""
"""
import random, sys
import time

from gmpy2 import mpz, powmod, invert, is_prime, random_state, mpz_urandomb, rint_round, log2, gcd, mpz_random, t_div

rand = random_state(random.randrange(sys.maxsize))


class PrivateKey(object):
    def __init__(self, p, q, n):
        if p == q:
            self.l = p * (p - 1)
        else:
            self.l = (p - 1) * (q - 1)
        try:
            self.m = invert(self.l, n)
        except ZeroDivisionError as e:
            print(e)
            exit()


class PublicKey(object):
    def __init__(self, n):
        self.n = n
        self.n_sq = n * n
        self.g = n + 1
        self.bits = mpz(rint_round(log2(self.n)))


def generate_prime(bits):
    """Will generate an integer of b bits that is prime using the gmpy2 library  """
    while True:
        possible = mpz(2) ** (bits - 1) + mpz_urandomb(rand, bits - 1)
        if is_prime(possible):
            return possible


def generate_keypair(bits):
    """ Will generate a pair of paillier keys bits>5"""
    p = generate_prime(bits // 2)
    # print(p)
    q = generate_prime(bits // 2)
    # print(q)
    n = p * q
    return PrivateKey(p, q, n), PublicKey(n)


def enc(pub, plain):  # (KeyPub key, plaintext) #to do
    G_of_m = powmod(pub.g, plain, pub.n_sq)  # 这里先计算G的m次方mod n的平方
    while True:
        r = mpz_random(rand, pub.n)
        if gcd(r, pub.n) == 1:  # 检查生成的r是否满足gcd=1
            break
    R_of_n = powmod(r, pub.n, pub.n_sq)  # 计算R的n次方mod n的平方
    cipher = powmod(G_of_m * R_of_n, 1, pub.n_sq)  # 计算密文c
    return cipher


def dec(priv, pub, cipher):  # (KeyPriv key, KeyPub key, cipher) #to do
    C_of_l = powmod(cipher, priv.l, pub.n_sq)  # C的l次方mod n的平方
    L_of_x = t_div(C_of_l - 1, pub.n)  # 计算L(x)
    plain = powmod(L_of_x * priv.m, 1, pub.n)  # 解密
    return plain


def enc_add(pub, m1, m2):  # to do
    """Add one encrypted integer to another"""
    return powmod(m1 * m2, 1, pub.n_sq)  # m1*m2 mod n^2


def enc_add_const(pub, m, c):  # to do
    """Add constant n to an encrypted integer"""
    G_of_c = powmod(pub.g, c, pub.n_sq)
    return powmod(m * G_of_c, 1, pub.n_sq)  # m*g^c mod n^2


def enc_mul_const(pub, m, c):  # to do
    """Multiplies an encrypted integer by a constant"""
    return powmod(m, c, pub.n_sq)  # m^c mod n^2


if __name__ == '__main__':
    priv, pub = generate_keypair(1024)

    time_start = time.time()
    c1 = enc(pub, 16)
    time_end = time.time()
    print("加密时间s：", time_end - time_start)
    time_start = time.time()
    p1 = dec(priv, pub, c1)
    time_end = time.time()
    print("解密时间s：", time_end - time_start)
    print(p1)


    time_start = time.time()
    c2 = enc(pub, 27)
    time_end = time.time()
    print("加密时间s：", time_end - time_start)
    time_start = time.time()
    p2 = dec(priv, pub, c2)
    time_end = time.time()
    print("解密时间s：", time_end - time_start)
    print(p2)


    time_start = time.time()
    c3 = enc_add(pub, c1, c2)
    time_end = time.time()
    print("加密时间s：", time_end - time_start)
    time_start = time.time()
    p3 = dec(priv, pub, c3)
    time_end = time.time()
    print("解密时间s：", time_end - time_start)
    print(p3)


    time_start = time.time()
    c4 = enc_add_const(pub, c1, 15)
    time_end = time.time()
    print("加密时间s：", time_end - time_start)
    time_start = time.time()
    p4 = dec(priv, pub, c4)
    time_end = time.time()
    print("解密时间s：", time_end - time_start)
    print(p4)


    time_start = time.time()
    c5 = enc_mul_const(pub, c1, 10)
    time_end = time.time()
    print("加密时间s：", time_end - time_start)
    time_start = time.time()
    p5 = dec(priv, pub, c5)
    time_end = time.time()
    print("解密时间s：", time_end - time_start)
    print(p5)

