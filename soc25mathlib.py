
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
    result = args[0]
    for i in range(1,len(args)):
        result = pair_gcd(result,args[i])
    return result    
    

def pair_lcm(a: int, b: int) -> int:
    # todo: return the lcm of a and b using formula lcm(a, b) = abs(a*b) // gcd(a, b)
    lcm = (a*b)//pair_gcd(a,b)
    return lcm

def lcm(*args: int) -> int:
    # todo: return the lcm of all arguments (assume 2+ args always)
    result = 1
    for i in range(len(args)):
        result *= pair_lcm(result,args[i])
    return result 

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
    result = 0
    N = 1
    for j in n : 
        N*=j
    for i in range(len(a)):
        x = N//n[i]
        result+= x*mod_inv(x,n[i])*a[i]
    return result%N

def is_quadratic_residue_prime(a: int, p: int) -> int:
    # todo: return 1 if a is QR mod p, -1 if QNR mod p, 0 if not coprime to p
    a = pow(a,(p-1)//2,p)
    if (a==p-1) :
        return -1
    return a

def is_quadratic_residue_prime_power(a: int, p: int, e: int) -> int:
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
