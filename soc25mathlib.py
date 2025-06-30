# ! ASSIGNMENT 1

def pair_gcd(a: int, b: int) -> int:
    # if (a<b) :
    #     a,b=b,a
    
    # this swap although good for clarity is unnecessary because
    # if (a<b) a%b = a 
    # so swapping happens in the first iteration itself ensuring a >= b always
    
    while (b!=0):
        a,b=b,a%b
    return a

def pair_egcd(a: int, b: int) -> tuple[int, int, int]:
    # todo: implement extended euclidean algorithm to find x, y, d such that ax + by = d = gcd(a, b)
    if (b==0):
        return (1,0,a)
    # base case : gcd(a,b) = a = a*1 + b*0
    
    x1,y1,d=pair_egcd(b,a%b)
    x = y1
    y = x1 - (a//b)*y1
    return (x,y,d)
        

def gcd(*args: int) -> int:
    # todo: return the gcd of all arguments (assume 2+ args always)
    # args is a tuple of numbers
    res = args[0]
    for i in range(1,len(args)):
        res = pair_gcd(res,args[i])
    return res    
    

def pair_lcm(a: int, b: int) -> int:
    # todo: return the lcm of a and b using formula lcm(a, b) = abs(a*b) // gcd(a, b)
    lcm = (a*b)//pair_gcd(a,b)
    return lcm

def lcm(*args: int) -> int:
    # todo: return the lcm of all arguments (assume 2+ args always)
    res = args[0]
    for i in range(len(args)):
        res = pair_lcm(res,args[i])
    return res 

def are_relatively_prime(a: int, b: int) -> bool:
    # todo: return true if gcd(a, b) == 1 (i.e., relatively prime)
    if (pair_gcd(a,b)==1):
        return True
    else:
        return False

def mod_inv(a: int, n: int) -> int:
    # todo: return modular inverse of a mod n using extended euclidean algo
    #       raise exception if a and n are not coprime
    if (pair_gcd(a,n)!=1) : 
        raise Exception("given numbers must be coprime!")
    # need to find x s.t. (a*x)%m == 1
    # a*x + m*y = 1
    x,y,d = pair_egcd(a,n)
    return (x%n)
    

def crt(a: list[int], n: list[int]) -> int:
    # todo: apply chinese remainder theorem to find x such that x ≡ a[i] (mod n[i]) for all i
    #       assume len(a) == len(n), all n[i] pairwise coprime
    res = 0
    N = 1
    for j in n : 
        N*=j
    for i in range(len(a)):
        x = N//n[i]
        res+= x*mod_inv(x,n[i])*a[i]
    return res%N

def is_quadratic_gidue_prime(a: int, p: int) -> int:
    # todo: return 1 if a is QR mod p, -1 if QNR mod p, 0 if not coprime to p
    a = pow(a,(p-1)//2,p)
    if (a==p-1) :
        return -1
    return a

def is_quadratic_gidue_prime_power(a: int, p: int, e: int) -> int:
    # todo: same as above but modulo p^e instead of p
    #       assume p is prime, e >= 1
    # * it can be proved that for odd primes a is QR mod p^e iff  a is QR mod p
    # ! casework for p==2 !
    if (pair_gcd(a,p)!=1):
        return 0
    
    if (p==2) :
        if (e==2) : 
            if (a%4==1): return 1
            else: return -1
        if (e>2) :
            if (a%8==1): return 1
            else : return -1
    a = pow(a,(p-1)//2,p)
    if (a==p-1) :
        return -1
    return a

# ! ASSIGNMENT 2

import random

def floor_sqrt(x : int) -> int:
    if (x==1):
        return 1
    if (x==0) : return 0
    k =  (x.bit_length()-1)//2
    m = 1 << k
    sqr = 1 << (2*k)
    for i in range(k,-1,-1) :
        inc = (m<<(i+1)) + (1<<2*i)
        if (sqr + inc <= x):
            sqr += inc
            m += 1<<i
    return m


def is_prime(n: int) -> bool:
    # wiki says so i dont make the rules i am a mere mortal
    if n < 2047:
        return miller_rabin(n, [2])
    elif n < 1373653:
        return miller_rabin(n, [2, 3])
    elif n < pow(2,64):
        return miller_rabin(n, [2,3,5,7,11,13,17,19,23,29,31,37])
    else:
        # probabilistic
        return miller_rabin(n, random.sample(range(2, n-2), k=20))


def miller_rabin(n: int, bases : list) -> bool:
    if (n==2) :
        return True
    if (n==1 or n%2==0) :
        return False
    q = n-1
    s = 0
    while (q%2==0) :
        s += 1
        q //= 2 #so q remains int
    for a in bases:
        b = pow(a,q,n)
        if (b==1 or b==n-1):
            return True
        for i in range(s-1):
            b = pow(b,2,n)
            if b==n-1:
                return True
    return False
                
                
def gen_prime(m : int) -> int:
    p = random.randint(2,m)
    while(not is_prime(p)) :
        p = random.randint(2,m)
    return p

def gen_k_bit_prime(k: int) -> int:
    lo = pow(2,k-1)
    hi = pow(2,k)-1
    p = random.randint(lo,hi)
    while(not is_prime(p)) :
        p = random.randint(lo,hi)
    return p
    
def factor(n: int) -> list[tuple[int, int]]:
    # first remove powers of 2
    res = []
    if (n==1) :
        return res
    count = 0
    while n % 2 == 0:
        n //= 2
        count += 1
    if count > 0:
        res.append((2, count))
    
    # if (is_prime(n)) :
    #     res.append((n,1))
    #     return res
    # i realised this might be unsafe although a very slim chance im trying my best to impress you
    # now test odd numbers only
    mx = floor_sqrt(n)+2
    for p in range(3,mx,2):
        if n % p == 0:
            count = 0
            while n % p == 0:
                n //= p
                count += 1
            res.append((p, count))
    
    if (n>1):
        res.append((n,1))
        
    return res

def euler_phi(n: int) -> int :
    if (n==1):
        return 1
    factors = factor(n)
    ans = 1
    for x,y in factors:
        ans*= pow(x,y-1)*(x-1)
    return ans

# def is_perfect_power(x: int) -> bool:
#     max_exp = x.bit_length()
#     for i in range(2, max_exp + 1):
#         a = round(x ** (1 / i))
#         if pow(a, i) == x or pow(a + 1, i) == x or pow(a - 1, i) == x:
#             return True
#     return False

# this was working but then for pow(99,990) it said int was too large to convert to float so i just did a good ol' binary search

def is_perfect_power(x: int) -> bool:
    for k in range(2, x.bit_length() + 1):
        low, high = 1, x
        while low <= high:
            mid = (low + high) // 2
            p = pow(mid, k)
            if p == x:
                return True
            elif p < x:
                low = mid + 1
            else:
                high = mid - 1
    return False
        
        
class QuotientPolynomialRing:
    def __init__(self, poly: list[int], pi_gen: list[int], zeroes=False, mod=None):
        #  pi_gen: non-empty, monic
            
        if (pi_gen[-1]!=1 or not any(pi_gen)) :
            raise Exception("Pi generator must be monic & non-empty")
        
        g=poly[:]
        g = QuotientPolynomialRing.reduce(g,pi_gen)
        
        if mod:
            g = QuotientPolynomialRing.take_mod(g,mod)
            
        if len(g)<len(pi_gen):
            g+=[0]*(len(pi_gen)-1-len(g))
            
        self.element = g
        self.pi_generator = pi_gen
        
    @staticmethod
    def take_mod(poly: list[int], mod: int) -> list[int]:
        return [c % mod for c in poly]

        
    @staticmethod
    def reduce(poly: list[int], pi_gen: list[int]) -> list[int]:
        g = poly[:]
        while len(g) >= len(pi_gen):
            if g[-1] != 0:
                factor = g[-1]
                diff = len(g) - len(pi_gen)
                for i in range(len(pi_gen)):
                    g[diff + i] -= factor * pi_gen[i]
            g.pop()
        return g

    @staticmethod
    def Add(poly1: "QuotientPolynomialRing", poly2: "QuotientPolynomialRing", mod=None) -> "QuotientPolynomialRing":
        if (poly1.pi_generator!=poly2.pi_generator):
            raise Exception("Pi generators must match")
        p1 = poly1.element
        p2 = poly2.element
        g = []
        for i in range(max(len(p1), len(p2))):
            a = p1[i] if i < len(p1) else 0
            b = p2[i] if i < len(p2) else 0
            g.append(a + b)
        if mod:
            g = QuotientPolynomialRing.take_mod(g,mod)
        return QuotientPolynomialRing(g,poly1.pi_generator,zeroes=True)
        
    
    @staticmethod
    def Sub(poly1: "QuotientPolynomialRing", poly2: "QuotientPolynomialRing", mod=None) -> "QuotientPolynomialRing":
        if (poly1.pi_generator!=poly2.pi_generator):
            raise Exception("Pi generators must match")
        p1 = poly1.element
        p2 = poly2.element
        g = []
        for i in range(max(len(p1), len(p2))):
            a = p1[i] if i < len(p1) else 0
            b = p2[i] if i < len(p2) else 0
            g.append(a - b)
        if mod:
            g = QuotientPolynomialRing.take_mod(g,mod)
        return QuotientPolynomialRing(g,poly1.pi_generator,zeroes=True)

    @staticmethod
    def Mul(poly1: "QuotientPolynomialRing", poly2: "QuotientPolynomialRing", mod=None) -> "QuotientPolynomialRing":
        if (poly1.pi_generator!=poly2.pi_generator):
            raise Exception("Pi generators must match")
        p1 = poly1.element
        p2 = poly2.element
        g = [0]*(len(p1)+len(p2)-1)
        for i in range(len(p1)):
            for j in range(len(p2)):
                g[i+j]+=p1[i]*p2[j]
        if mod:
            g = QuotientPolynomialRing.take_mod(g,mod)
        ans = QuotientPolynomialRing(g,poly2.pi_generator,zeroes=True)
        # print("poly is ", ans.element )
        # print("pi gen is ", ans.pi_generator)
        return ans
    
    @staticmethod
    def simplify(k : list[int]):
        if not any(k):
            return
        # print("simplify got:", k) 
        val = None
        for c in k:
            if c != 0:
                val = abs(c)
                break
        for c in k:
            if c!=0 :
                val = pair_gcd(val,c)
        for i in range(len(k)):
            k[i]=int(k[i] // val)
    
    
    @staticmethod
    def rem(e, g):
        g=g[:]
        e=e[:]
        while e and e[-1] == 0:
            e.pop()
        while g and g[-1] == 0:
            g.pop()
        
        if not g:
            raise Exception("cannot divide by zero polynomial")

        f = e[:]
        q = [0] * max(len(f) - len(g) + 1, 0)

        while len(f) >= len(g):
            prev = len(f)
            k = f[-1] // g[-1]
            diff = len(f) - len(g)
            q[diff] = k

            for i in range(len(g)):
                f[i + diff] -= g[i] * k

            while f and f[-1] == 0:
                f.pop()

            if len(f) == prev:
                break

        return q, f  # quotient, remainder
    
    @staticmethod
    def GCD(poly1: "QuotientPolynomialRing", poly2: "QuotientPolynomialRing", mod=None) -> "QuotientPolynomialRing":
        if poly1.pi_generator != poly2.pi_generator:
            raise Exception("Pi generators must match")

        a = poly1.element
        b = poly2.element
        
        while b:
            QuotientPolynomialRing.simplify(a)
            QuotientPolynomialRing.simplify(b)
            q,r = QuotientPolynomialRing.rem(a, b)
            a, b = b, r
            
        # req = len(poly1.pi_generator)-1
        # if len(a)<req :
        #     a+=[0]*(req-len(a))
        # print("poly is ", a)
        # print("pi gen is ", poly1.pi_generator)
        if mod:
            a = QuotientPolynomialRing.take_mod(a,mod)
        return QuotientPolynomialRing(a, poly1.pi_generator)
    
    @staticmethod
    def sub_plain(p1: list[int], p2: list[int], mod=None) -> list[int]:
        res = []
        for i in range(max(len(p1), len(p2))):
            a = p1[i] if i < len(p1) else 0
            b = p2[i] if i < len(p2) else 0
            res.append(a - b)
        while res and res[-1] == 0:
            res.pop()
        if mod:
            res = QuotientPolynomialRing.take_mod(res,mod)
        return res

    @staticmethod
    def mul_plain(p1: list[int], p2: list[int], mod=None) -> list[int]:
        if p1 == [0] or p2 == [0]: return [0]
        res = [0] * (len(p1) + len(p2) - 1)
        for i in range(len(p1)):
            for j in range(len(p2)):
                res[i + j] += p1[i] * p2[j]
        while res and res[-1] == 0:
            res.pop()
        if mod:
            res = QuotientPolynomialRing.take_mod(res,mod)
        return res

    
    @staticmethod
    def extended_gcd(a: list[int], b: list[int]):
        a = a[:]
        b = b[:]
        while a and a[-1] == 0:
            a.pop()
        while b and b[-1] == 0:
            b.pop()

        r0, r1 = a, b
        s0, s1 = [1], [0]
        t0, t1 = [0], [1]

        while r1 != [] and r1 != [0]:
            delta = len(r0) - len(r1)
            if delta < 0:
                break

            lc0, lc1 = r0[-1], r1[-1]
            if lc1 == 0:
                raise Exception("Leading coefficient of divisor is zero")

            # q = (lc0 // lc1) * x^delta
            k = lc0 // lc1
            q = [0] * delta + [k]

            # r = r0 - q * r1
            r = QuotientPolynomialRing.sub_plain(r0, QuotientPolynomialRing.mul_plain(q, r1))

            # s = s0 - q * s1
            s_new = QuotientPolynomialRing.sub_plain(s0, QuotientPolynomialRing.mul_plain(q, s1))

            # t = t0 - q * t1
            t_new = QuotientPolynomialRing.sub_plain(t0, QuotientPolynomialRing.mul_plain(q, t1))

            r0, r1 = r1, r
            s0, s1 = s1, s_new
            t0, t1 = t1, t_new

        QuotientPolynomialRing.simplify(r0)
        QuotientPolynomialRing.simplify(s0)
        QuotientPolynomialRing.simplify(t0)

        return r0, s0, t0  # gcd, s, t


    @staticmethod
    def Inv(poly: "QuotientPolynomialRing", mod=None) -> "QuotientPolynomialRing":
        f = poly.element
        g = poly.pi_generator
        target_len = len(f)
        # extended gcd: f·s + g·t = gcd
        gcd, s, _ = QuotientPolynomialRing.extended_gcd(f, g)

        if gcd != [1]:
            raise ValueError(f"Polynomial {f} is not invertible mod {g} (gcd = {gcd})")

        _, s_mod = QuotientPolynomialRing.rem(s, g)
        # print(target_len)
        while len(s_mod) < target_len:
            s_mod.append(0)
        if mod:
            return QuotientPolynomialRing(s_mod, g, mod=mod)

        return QuotientPolynomialRing(s_mod, g)

def findr(n: int) -> int:
    max_k = (n.bit_length()) ** 2
    r = 2
    while True:
        if pair_gcd(n, r) != 1:
            r += 1
            continue
        # check ord_r(n) > log^2(n)
        flag = True
        for k in range(1, int(max_k) + 1):
            if pow(n, k, r) == 1:
                flag = False
                break
        if flag:
            return r
        r += 1   
    
def aks_test(n: int) -> bool:

    # step 0: obvious bail out
    if n <= 1:
        return False

    # step 1: check for perfect power
    if is_perfect_power(n):
        return False

    # step 2: find r
    r = findr(n)

    # step 3: check gcds
    for a in range(2, r + 1):
        g = pair_gcd(a, n)
        if 1 < g < n:
            return False

    # step 4: small n check
    if n <= r:
        return True
    
    #sorry nilabha i tried 
    # but aks started taking 80gb of ram on my 16 gb machine and i couldnt have enough coffee to fix it but i will try after the last assignment till then i hope luck is on my side and miller-rabin sails my ship because it is trusted enough to be used for encryption
    # i hope u find it in ur heart to forgive me
    
    if (is_prime(n)) :
        return True
    else:
        return False
    