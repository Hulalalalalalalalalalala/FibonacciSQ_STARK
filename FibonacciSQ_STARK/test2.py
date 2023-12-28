from FibonacciSQ_STARK.Constraints import part1
from FibonacciSQ_STARK.Stark_polynomial import X
from FibonacciSQ_STARK.channel import Channel
from FibonacciSQ_STARK.merkle import MerkleTree

a, g, G, h, H, eval_domain, f, f_eval, f_merkle, channel = part1()
print('Success!')

numer0 = f - 1
denom0 = X - 1
p0 = numer0 / denom0

numer1 = f - 2338775057
denom1 = X - g**1022
p1 = numer1 / denom1

numer2 = f(g**2 * X) - f(g * X)**2 - f**2
denom2 = (X**1024 - 1) / ((X - g**1021) * (X - g**1022) * (X - g**1023))

p2 = numer2 / denom2

print('deg p0 =', p0.degree())
print('deg p1 =', p1.degree())
print('deg p2 =', p2.degree())

def get_CP(channel):
    alpha0 = channel.receive_random_field_element()
    alpha1 = channel.receive_random_field_element()
    alpha2 = channel.receive_random_field_element()
    return alpha0*p0 + alpha1*p1 + alpha2*p2

test_channel = Channel()
CP_test = get_CP(test_channel)
assert CP_test.degree() == 1023, f'The degree of cp is {CP_test.degree()} when it should be 1023.'
assert CP_test(2439804) == 838767343, f'cp(2439804) = {CP_test(2439804)}, when it should be 838767343'
print('Success!')

def CP_eval(channel):
    CP = get_CP(channel)
    return [CP(d) for d in eval_domain]

channel = Channel()
CP_merkle = MerkleTree(CP_eval(channel))
channel.send(CP_merkle.root)

assert CP_merkle.root == 'a8c87ef9764af3fa005a1a2cf3ec8db50e754ccb655be7597ead15ed4a9110f1', 'Merkle tree root is wrong.'
print('Success!')