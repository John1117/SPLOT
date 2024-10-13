# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({'font.size': 15,
                     'figure.figsize': [8, 8]})


# %%
# Normal distr
a1 = 0.0705230784
a2 = 0.0422820123
a3 = 0.0092705272
a4 = 0.0001520143
a5 = 0.0002765672
a6 = 0.0000430638
half_approx_err_fn = lambda x: (1 - 1 / (1 + a1 * x + a2 * x ** 2 + a3 * x ** 3 + a4 * x ** 4 + a5 * x ** 5 + a6 * x ** 6) ** 16)
approx_err_fn = lambda x: half_approx_err_fn(np.abs(x)) * np.sign(x)  # Approx err fn
normal_CDF = lambda x, u=0, s=1: 0.5 * (1 + approx_err_fn((x - u) / s / 2 ** 0.5))  # CDF of normal distr
normal_PDF = lambda x, u=0, s=1: np.exp(-0.5 * ((x - u) / s) ** 2) / s / (2 * np.pi) ** 0.5  # PDF of normal distr

x = np.linspace(-5, 5, 100)
plt.plot(x, normal_PDF(x), 'k', label='PDF')
plt.plot(x, normal_CDF(x), 'b', label='CDF')
plt.title('PDF and CDF of normal distribution')
plt.xlabel('X')
plt.ylabel('f(X)')
plt.legend()
plt.show()


# %%
s0 = 1   # scale of std
std_fn = lambda pu: s0 * pu * (1 - pu)  # Assumed prob std fn

A_fn = lambda pu: 1 / (normal_CDF(1, pu, std_fn(pu)) - normal_CDF(0, pu, std_fn(pu)))  # Scaling fn of normal distr
phi_fn = lambda p, pu=0.6: A_fn(pu) * normal_PDF(p, pu, std_fn(pu))
PHI_fn = lambda p, pu=0.6: A_fn(pu) * normal_CDF(p, pu, std_fn(pu))


p = np.linspace(0, 1, 1000)
for pu in np.linspace(0.05, 0.95, 10):
    plt.plot(p, phi_fn(p, pu), c=(1-pu, 0, pu), label=f'pu = {round(pu, 1)}')

plt.title("PDF of player's believed probability")
plt.xlabel("Player's believed probability")
plt.ylabel('Probability density')
#plt.legend()
plt.show()

# %%
m = [-0.1, 3]
pT = 0.45
n_pt = 1000
p = np.linspace(0, 1, n_pt)
gpT = np.linspace(pT, 1, int(n_pt*pT))

plt.figure(figsize=(10, 10))
plt.plot(p, phi_fn(p, 0.5), 'g', label='Pool-1')
plt.fill_between(gpT, 0, phi_fn(gpT, 0.5), color='g', alpha=0.2)

plt.plot(p, phi_fn(p, 0.25), 'b', label='Pool-2')
plt.fill_between(gpT, 0, phi_fn(gpT, 0.25), color='b', alpha=0.2)

plt.plot([pT, pT], m, 'k--', label='1 / banker-odd', alpha=0.2)
plt.plot([0, 0], m, 'k-', alpha=0.2)
plt.plot([1, 1], m, 'k-', alpha=0.2)
plt.ylim(m)
plt.title("PDF of players' believed prob")
plt.xlabel("Player's believed prob")
plt.ylabel('PDF')
plt.legend()
plt.show()

# %%
pt = 0.3
r_fn = lambda b, pu: PHI_fn(1, pu) - PHI_fn(1/b, pu)
R_fn = lambda b, pu, pt: r_fn(b, pu) * (1 - pt * b)

pu = 0.7
pb = np.linspace(0.1, 0.9)
b = 1/pb
r = r_fn(b, pu)
R = R_fn(b, pu, pt)

my = [-2.2, 1.2]
mx = [0, 1]
plt.plot([pt, pt], my, 'k--', label='True prob', alpha=0.2)
plt.plot(mx, [0, 0], 'k', alpha=0.2)
plt.plot(pb, 1-pt*b, 'b--', label='Not considering bet ratio')
plt.plot(pb, R, 'b', label='Considering bet ratio')
plt.plot(pb[R.argmax()], R.max(), 'ro', label='Max')

plt.legend(loc=8)
plt.ylim(my)
plt.xlim(mx)

plt.yticks(color='b')
plt.xlabel('1 / banker-odd')
plt.ylabel('Expected return of banker', color='b')

plt.twinx()
plt.plot(pb, r, 'g-')
plt.ylabel('Bet ratio of player pool', color='g')
plt.yticks(color='g')
plt.show()


# %%
pt = 0.5
n_pt = 300
pu = np.linspace(0.1, 0.9, n_pt)
pb = np.linspace(0.1, 0.9, n_pt)
pu_m, pb_m = np.meshgrid(pu, pb)
R = R_fn(1/pb_m, pu_m, pt)
m = np.abs(R).max()

plt.figure(figsize=(9, 8))
plt.contourf(pb_m, pu_m, R, cmap='bwr', levels=n_pt, vmax=m, vmin=-m)
plt.plot(pb[R.argmax(axis=0)], pu, 'r', label='1 / optimal-odd')
plt.plot([pt, pt], [0.1, 0.9], 'k--', label='True prob', alpha=0.2)
plt.plot([0.1, 0.9], [pt, pt], 'k--', alpha=0.2)
plt.colorbar()
p_tk = np.round(np.linspace(0.1, 0.9, 5), 1)
plt.xticks(p_tk, p_tk)
plt.yticks(p_tk, p_tk)
plt.title(f'Expected return of banker', color='b')
plt.ylabel('Mean believed prob of players')
plt.xlabel("1 / banker-odd")
plt.legend()
plt.show()


# %%
n_pt = 1000
d = 1e-2
pt = np.linspace(d, 1-d, n_pt).reshape(1, -1)
pu = pt
pb = np.linspace(d, 1-d, n_pt).reshape(-1, 1)
R = R_fn(1/pb, pu, pt)
opt_pb = pb[R.argmax(axis=0)].squeeze()
opt_R = R.max(axis=0)
opt_r = r_fn(1/opt_pb, pu).squeeze()

pt = pt.squeeze()
plt.plot(0.5, 0, 'k--', label='Uniform (max-exploited)')
plt.plot(0.5, 0, 'k-.', label='Delta (min-exploited)')

a = 0.25
plt.plot(pt, opt_pb, 'r')
plt.plot(pt, pt, 'r-.', alpha=a)
plt.plot(pt, pt**0.5, 'r--', alpha=a)
plt.legend(title="Players' PDF")
p_tk = np.round(np.linspace(0.1, 0.9, 5), 1)
plt.xticks(p_tk, p_tk)
plt.yticks(p_tk, p_tk)
plt.xlabel('True prob')
plt.ylabel('1 / optimal-odd', color='r')
plt.yticks(color='r')


plt.twinx()
plt.plot(pt, opt_R, 'b')
plt.plot(pt, np.zeros_like(pt), 'b-.', alpha=a)
plt.plot(pt, (1 - pt**0.5)**2, 'b--', alpha=a)
plt.ylabel('Expected return of banker', color='b')
plt.yticks(color='b')

plt.show()

# %%