# A Beginner's Guide to the Discrete Fourier Transform and Spectral Methods

### For readers of: *Optimal Mixing of Passive Scalars — Python Implementation and Numerical Results*

---

## Why This Guide Exists

The Numerical Methods section (Section 3) of the optimal mixing report uses concepts from Fourier analysis and spectral methods — tools from signal processing and applied mathematics that can feel very abstract at first. This guide explains every concept from the ground up.

**If you have studied calculus and basic linear algebra, you have everything you need to follow this guide.**

---

## Part 1: The Big Picture — What Is a Spectral Method?

### 1.1 The Problem We Want to Solve

The mixing simulation involves solving a partial differential equation (PDE):

$$\frac{\partial \theta}{\partial t} + u \cdot \nabla \theta = 0$$

This describes how a scalar field $\theta(x, y, t)$ — think of it as the concentration of ink in water — changes over time under a velocity field $u$. The term $\nabla \theta$ means "gradient of $\theta$": a vector pointing in the direction of steepest increase.

To solve this on a computer, we must *discretise* it: approximate it on a finite grid of points. There are many ways to do this. The **spectral method** is one of the most accurate.

### 1.2 Two Ways to Represent a Function

Imagine a wiggly function $f(x)$ defined on $[0, 1]$:

**Way 1 — Physical space:** Store the values $f(x_0), f(x_1), \ldots, f(x_{N-1})$ at grid points. Think of it as a bar chart of heights.

**Way 2 — Frequency space (Fourier space):** Represent $f$ as a sum of sine and cosine waves of different frequencies. Instead of storing heights, store *how much of each frequency is present*.

These are just two descriptions of the same function — like describing a song by its audio waveform vs. its sheet music. You can switch between them freely.

**The key insight:** Some operations that are hard in physical space (like computing derivatives) become trivially simple in frequency space — just multiply by a number.

---

## Part 2: Waves and Frequencies

### 2.1 What Is a Frequency?

A pure wave of frequency $k$ on $[0, 1]$ looks like:

$$f_k(x) = \sin(2\pi k x) \qquad \text{or} \qquad f_k(x) = \cos(2\pi k x)$$

The number $k$ counts how many complete oscillations fit inside $[0, 1]$:

- $k = 1$: one full oscillation — a single gentle curve
- $k = 2$: two full oscillations — a slightly wigglier curve
- $k = 10$: ten oscillations — a rapidly oscillating wave

Higher $k$ = higher frequency = finer (smaller-scale) features.

### 2.2 Any Periodic Function Is a Sum of Waves

This is the central claim of Fourier analysis: **any periodic function can be built by adding up sine and cosine waves**. This sum is the Fourier series:

$$f(x) = a_0 + \sum_{k=1}^{\infty} \left[ a_k \cos(2\pi k x) + b_k \sin(2\pi k x) \right]$$

The coefficients $a_k$ and $b_k$ tell you how much of each frequency is present.

**A concrete example:** A square wave (alternating $+1$ and $-1$) can be written as:

$$f(x) = \frac{4}{\pi}\left[\sin(2\pi x) + \frac{1}{3}\sin(6\pi x) + \frac{1}{5}\sin(10\pi x) + \cdots\right]$$

Adding the first few terms already gives a recognisable approximation; the infinite series is exact.

### 2.3 Complex Exponentials — A Compact Notation

In practice, instead of writing $\sin$ and $\cos$ separately, mathematicians use **complex exponentials**, using Euler's formula:

$$e^{2\pi i k x} = \cos(2\pi k x) + i\,\sin(2\pi k x)$$

The Fourier series then takes the symmetric form:

$$f(x) = \sum_{k=-\infty}^{\infty} \hat{f}(k)\, e^{2\pi i k x}$$

The complex number $\hat{f}(k)$ is the **Fourier coefficient** at wavenumber $k$. Its magnitude $|\hat{f}(k)|$ is the amplitude of that frequency, and its argument (angle) $\arg(\hat{f}(k))$ is the phase shift.

---

## Part 3: The Discrete Fourier Transform (DFT)

### 3.1 From Continuous to Discrete

The Fourier series works for continuous functions. Computers, however, work with finite lists of numbers. We need a discrete version.

Suppose we have $N$ equally spaced sample points:

$$x_j = \frac{j}{N}, \quad j = 0, 1, 2, \ldots, N-1$$

The **Discrete Fourier Transform (DFT)** takes the $N$ values $f[0], f[1], \ldots, f[N-1]$ and produces $N$ Fourier coefficients:

$$\hat{f}[k] = \sum_{j=0}^{N-1} f[j]\, e^{-2\pi i j k / N}, \quad k = 0, 1, \ldots, N-1$$

The **Inverse DFT (IDFT)** goes back:

$$f[j] = \frac{1}{N} \sum_{k=0}^{N-1} \hat{f}[k]\, e^{2\pi i j k / N}$$

**No information is lost.** Going forward and back is exact. The DFT and IDFT are just two different representations of the same data — like converting between Celsius and Fahrenheit.

### 3.2 What Do the Fourier Coefficients Mean?

$\hat{f}[k]$ is a complex number. Its:
- **Magnitude** $|\hat{f}[k]|$ tells you how much of frequency $k$ is present in $f$
- **Phase** $\arg(\hat{f}[k])$ tells you how that wave is shifted (its offset)

If $f$ is real-valued (as physical quantities always are), the negative-frequency coefficients are redundant: $\hat{f}[-k] = \overline{\hat{f}[k]}$ (complex conjugate).

### 3.3 The FFT — The Practical Tool

Computing the DFT directly requires $O(N^2)$ operations — for $N = 1000$, that's $10^6$ multiplications. The **Fast Fourier Transform (FFT)** algorithm computes the *exact same result* in $O(N \log N)$ operations — for $N = 1000$, that's only $\approx 10{,}000$.

FFT vs. DFT: the algorithm is different, the result is identical.

In NumPy:
```python
f_hat = np.fft.fft(f)       # 1D FFT
f_back = np.fft.ifft(f_hat) # 1D inverse FFT

F_hat = np.fft.fft2(F)      # 2D FFT (applied to each axis)
F_back = np.fft.ifft2(F_hat)# 2D inverse FFT
```

### 3.4 The 2D DFT

For a 2D function $f(x, y)$ defined on an $N \times N$ grid, the DFT is applied in both directions independently:

$$\hat{f}[k_y, k_x] = \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} f[i, j]\, e^{-2\pi i (i k_y + j k_x)/N}$$

**Index convention used in this project (and NumPy):**
- Row index $i$ corresponds to the **$y$-direction**
- Column index $j$ corresponds to the **$x$-direction**
- So `theta_hat[ky, kx]` is the coefficient for wavenumber $(k_x, k_y)$

This is why `DEL_X` has shape `(1, N)` (acts on columns) and `DEL_Y` has shape `(N, 1)` (acts on rows).

---

## Part 4: Wavenumbers — What Are They?

### 4.1 The Wavenumber Array

After computing `fft(f)` on an array of length $N$, the output has $N$ entries corresponding to $N$ different frequencies. But they are stored in a specific order.

For $N = 8$, the array indices and their corresponding effective wavenumbers are:

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|-------|---|---|---|---|-----|---|---|---|
| Wavenumber | 0 | 1 | 2 | 3 | ±4 | −3 | −2 | −1 |

- **Index 0**: the constant (DC) component, $k = 0$ — the average value of $f$
- **Indices 1 to $N/2-1$**: positive frequencies
- **Index $N/2$**: the Nyquist frequency $k = N/2$ (special case, see below)
- **Indices $N/2+1$ to $N-1$**: negative frequencies

In the code:
```python
k     = np.arange(N)
k_eff = k - N * (k > N // 2)   # convert index → effective wavenumber
```

This is called "centring" the wavenumbers. To visualise a 2D spectrum with zero in the middle, use `np.fft.fftshift`.

### 4.2 Why Negative Frequencies?

Negative frequencies arise naturally from the complex exponential formulation. For a real-valued function $f$, the $k$ and $-k$ modes always come in complex-conjugate pairs. Together they produce a real-valued signal:

$$\hat{f}[k]\, e^{2\pi i k x} + \hat{f}[-k]\, e^{-2\pi i k x} = 2|\hat{f}[k]|\cos(2\pi k x + \phi)$$

So negative frequencies just represent the "mirror image" of positive frequencies — they are not physically different, just part of the mathematical convention.

### 4.3 The Nyquist Frequency — The Highest Representable Frequency

The grid has spacing $dx = 1/N$. The finest feature you can represent is a wave that alternates sign at every grid point — one oscillation per $2dx$. That frequency is:

$$k_{\text{Nyquist}} = \frac{N}{2}$$

**You cannot represent frequencies higher than $N/2$ on a grid with $N$ points.** If a function has fine-scale features at $k > N/2$, the DFT *cannot see them* — they get wrapped around and appear as lower-frequency noise. This is called **aliasing**.

**A real-world analogy:** A movie camera shoots at 24 frames per second. A car wheel spinning at more than 12 revolutions per second appears to spin backwards in the footage — this is aliasing. The camera can't represent frequencies faster than 12 Hz.

### 4.4 The Nyquist Mode and Why It Is Zeroed for First Derivatives

At $k = N/2$, the wave $e^{2\pi i (N/2) x}$ sampled at grid points $x_j = j/N$ gives:
$$e^{2\pi i (N/2)(j/N)} = e^{\pi i j} = (-1)^j$$

This alternates $+1, -1, +1, -1, \ldots$ — the same as $e^{-2\pi i (N/2) x}$. So $k = +N/2$ and $k = -N/2$ are **identical on the grid** — they are ambiguous.

For the **Laplacian** (second derivative), this is fine: $(-k)^2 = k^2$, so it does not matter which sign we choose.

For **first derivatives**, the sign matters: $+k$ and $-k$ give derivatives with opposite signs. Since the mode is ambiguous, we cannot assign a meaningful derivative to it. The safe choice is to zero it out:

```python
k_d = k_eff.copy()
k_d[N//2] = 0.0   # zero the Nyquist mode for first derivatives
```

`DEL_X` and `DEL_Y` use `k_d`; `LAP_INV` uses `k_eff`.

---

## Part 5: Spectral Differentiation — The Core Magic Trick

### 5.1 Why Differentiation in Physical Space Becomes Multiplication in Fourier Space

This is the single most important idea in spectral methods. Consider a function written as a sum of waves:

$$f(x) = \sum_k \hat{f}(k)\, e^{2\pi i k x}$$

Take the derivative term by term:

$$\frac{df}{dx} = \sum_k \hat{f}(k) \cdot \frac{d}{dx}\left[e^{2\pi i k x}\right] = \sum_k \hat{f}(k) \cdot (2\pi i k)\, e^{2\pi i k x}$$

The derivative of $e^{2\pi i k x}$ is just $(2\pi i k)\, e^{2\pi i k x}$.

**In Fourier space, taking a derivative simply means multiplying each coefficient by $2\pi i k$.**

Recipe for computing $\frac{df}{dx}$ spectrally:
1. Compute $\hat{f} = \text{FFT}(f)$
2. Multiply: $\widehat{df/dx}[k] = (2\pi i k)\, \hat{f}[k]$
3. Transform back: $df/dx = \text{IFFT}(\widehat{df/dx})$

```python
f_hat  = np.fft.fft(f)
df_hat = (2j * np.pi * k_d) * f_hat     # k_d has Nyquist zeroed
df_dx  = np.real(np.fft.ifft(df_hat))   # real part (discard tiny numerical imaginary part)
```

**Accuracy:** Spectral differentiation is **spectrally accurate** — the error decreases faster than any polynomial in $N$ as $N \to \infty$ (for smooth periodic functions). It is far more accurate than finite differences, which are only $O(dx^p)$ accurate for some fixed order $p$.

### 5.2 Second Derivatives and the Laplacian

Two derivatives each contribute a factor of $2\pi i k$:

$$(2\pi i k)^2 = -4\pi^2 k^2$$

So the second derivative multiplier is $-4\pi^2 k^2$ (real and negative).

In 2D, the Laplacian $\Delta f = \partial_{xx} f + \partial_{yy} f$ becomes:

$$\widehat{\Delta f}[k_y, k_x] = \left[-(2\pi k_x)^2 - (2\pi k_y)^2\right] \hat{f}[k_y, k_x] = -4\pi^2(k_x^2 + k_y^2)\, \hat{f}[k_y, k_x]$$

### 5.3 The Operators `DEL_X`, `DEL_Y` in the Code

```python
DEL_X = (2j * np.pi * k_d)[np.newaxis, :]   # shape (1, N)
DEL_Y = (2j * np.pi * k_d)[:, np.newaxis]   # shape (N, 1)
```

For a 2D field $\theta$:
- `DEL_X * theta_hat` multiplies each *column* (x-direction) by the corresponding $2\pi i k_x$ — giving $\widehat{\partial_x \theta}$
- `DEL_Y * theta_hat` multiplies each *row* (y-direction) by $2\pi i k_y$ — giving $\widehat{\partial_y \theta}$

**Why shapes $(1, N)$ and $(N, 1)$?** NumPy broadcasting means `(1,N)` is automatically repeated along rows to become `(N,N)` when multiplied with a `(N,N)` array. This saves memory without storing any redundant data.

---

## Part 6: The Laplacian and Its Inverse

### 6.1 `LAP_INV` — The Inverse Laplacian

We established that the Laplacian in Fourier space is multiplication by $-4\pi^2 |k|^2$. The **inverse Laplacian** is therefore division by the same factor:

$$\text{LAP\_INV}[k_y, k_x] = \frac{-1}{(2\pi)^2(k_x^2 + k_y^2)}, \quad (k_x, k_y) \neq (0, 0)$$

$$\text{LAP\_INV}[0, 0] = 0 \quad \text{(DC mode set to zero)}$$

To apply $\Delta^{-1}$ to a field $\theta$:
1. Compute $\hat{\theta} = \text{FFT2}(\theta)$
2. Multiply: $\widehat{\Delta^{-1}\theta} = \text{LAP\_INV} \cdot \hat{\theta}$
3. Transform back: $\Delta^{-1}\theta = \text{real}(\text{IFFT2}(\widehat{\Delta^{-1}\theta}))$

**Why zero the DC mode?** At $k = (0,0)$, we would be dividing by zero. The DC mode represents the spatial mean $\langle \theta \rangle = \int \theta\, dx\, dy$. This mean is conserved and decoupled from the rest of the dynamics, so we simply set it to zero.

**Physical interpretation of $\Delta^{-1}$:** The inverse Laplacian **smooths** a function. High-frequency (small-scale) features are suppressed by the large $|k|^2$ in the denominator. If $\theta$ is a concentrated blob, $\Delta^{-1}\theta$ is a broad, smooth distribution.

### 6.2 Why We Need $\Delta^{-1}$ for the Velocity

In the LTD optimal velocity formula:

$$v = -\Delta^{-1} P(\theta\,\nabla\Delta^{-1}\theta)$$

The $\Delta^{-1}$ appears **twice**:
1. The inner $\Delta^{-1}\theta$ converts the scalar field into a stream-function-like field (smoother, large-scale)
2. The outer $-\Delta^{-1}$ converts the nonlinear term into the velocity itself

This ensures the velocity has the right structure (divergence-free, with controlled enstrophy).

---

## Part 7: Parseval's Identity — Connecting Norms

### 7.1 The Statement

For a 1D DFT with $N$ points:

$$\sum_{j=0}^{N-1} |f[j]|^2 = \frac{1}{N} \sum_{k=0}^{N-1} |\hat{f}[k]|^2$$

In the NumPy convention (non-unitary), this becomes $\|\hat{f}\|^2 = N \|f\|^2$, so:

$$\|f\|_{L^2} \approx \frac{\|\hat{f}\|_{\ell^2}}{N}$$

In 2D (with $N^2$ points):

$$\|f\|_{L^2} \approx \frac{\|\hat{f}\|_{\ell^2}}{N^2}$$

### 7.2 What It Means

Parseval's identity says: **the total "energy" (sum of squares) is the same in physical space and Fourier space**, up to a normalisation factor.

This is analogous to the Pythagorean theorem: if you rotate a vector, its length doesn't change. The DFT is a rotation in an $N$-dimensional space, so it preserves lengths.

### 7.3 How It Is Used in the Code

To compute $\|\nabla v\|_{L^2}$ (needed for the enstrophy normalisation), the code computes in Fourier space:

```python
grad_v_norm = (1/N**2) * np.sqrt(
    np.sum(np.abs(DEL_X * v_x_hat)**2) +
    np.sum(np.abs(DEL_Y * v_x_hat)**2) +
    np.sum(np.abs(DEL_X * v_y_hat)**2) +
    np.sum(np.abs(DEL_Y * v_y_hat)**2)
)
```

Each `DEL_X * v_x_hat` is $\widehat{\partial_x v_x}$ in Fourier space, and the Parseval factor $1/N^2$ converts the Fourier-space norm to the physical-space $L^2$ norm.

---

## Part 8: The $H^{-1}$ Mix Norm — Measuring How Well-Mixed the Scalar Is

### 8.1 Why Not Just Use the $L^2$ Norm?

Under the passive scalar equation $\partial_t \theta + u \cdot \nabla \theta = 0$ with a divergence-free $u$, all $L^p$ norms are **conserved** — they never change. In particular:

$$\|\theta(t)\|_{L^2} = \|\theta_0\|_{L^2} \quad \text{for all } t$$

So the $L^2$ norm cannot tell us anything about mixing — it's constant regardless of how well the dye has been stirred.

We need a norm that *decreases* as the scalar gets mixed into finer and finer structures.

### 8.2 The $H^{-1}$ Norm

The **$H^{-1}$ norm** is defined as:

$$\|\theta\|_{H^{-1}}^2 = \sum_{k \neq 0} \frac{|\hat{\theta}(k)|^2}{4\pi^2 |k|^2}$$

Compare with the $L^2$ norm: $\|\theta\|_{L^2}^2 = \sum_k |\hat{\theta}(k)|^2$.

The difference: the $H^{-1}$ norm divides each mode's contribution by $|k|^2$, giving **much less weight to high-frequency (fine-scale) content**.

**What happens when we stir?** Initially, the scalar's energy is concentrated at low wavenumbers (smooth, large blobs). As the velocity stirs it, energy is transferred to higher and higher wavenumbers (finer filaments). The $H^{-1}$ norm penalises low-$k$ content heavily — so as energy moves to high $k$, the norm decreases.

### 8.3 Intuition for the $H^{-1}$ Norm

Think of it this way:
- A scalar with all its energy at low $k$ (one big blob) has **large** $H^{-1}$ norm — it is unmixed.
- A scalar with its energy spread uniformly across many $k$ values has **small** $H^{-1}$ norm — it is well-mixed.
- A perfectly uniform scalar ($\theta = $ const, all energy at $k=0$) has zero $H^{-1}$ norm — perfectly mixed.

### 8.4 The $\lambda^{-1}$ Weight

In code, the mix norm is computed via `LAMBDA_INV`:

$$\text{LAMBDA\_INV}[k] = \frac{1}{2\pi|k|} = \sqrt{-\text{LAP\_INV}[k]}$$

$$\|\theta\|_{H^{-1}}^2 = \frac{\|\text{LAMBDA\_INV} \cdot \hat{\theta}\|_{\ell^2}^2}{N^4}$$

```python
LAMBDA_INV = np.sqrt(np.where(LAP_INV != 0, -LAP_INV, 0.0))
mix_norm_sq = np.sum(np.abs(LAMBDA_INV * theta_hat)**2) / N**4
```

---

## Part 9: The Leray Projection — Enforcing Incompressibility

### 9.1 What Is Incompressibility?

The velocity field $u = (u_x, u_y)$ must satisfy $\nabla \cdot u = 0$, meaning:

$$\frac{\partial u_x}{\partial x} + \frac{\partial u_y}{\partial y} = 0$$

This is the **incompressibility condition**: fluid is not created or destroyed anywhere. In physical terms, any fluid flowing into a region must flow out equally.

### 9.2 The Helmholtz Decomposition

Any vector field $g$ can be split into two parts:

$$g = \underbrace{g_{\text{div-free}}}_{\nabla \cdot g_{\text{div-free}} = 0} + \underbrace{\nabla \phi}_{\text{pure gradient}}$$

where $\phi$ is some scalar potential. The **Leray projection** extracts just the divergence-free part:

$$Pg = g - \nabla \phi, \qquad \phi = \Delta^{-1}(\nabla \cdot g)$$

**Step by step:**
1. Compute the divergence: $D = \nabla \cdot g = \partial_x g_x + \partial_y g_y$
2. Solve $\Delta \phi = D$ (i.e., $\phi = \Delta^{-1} D$)
3. Subtract the gradient: $Pg = g - \nabla \phi$

### 9.3 In Fourier Space — It's Just Multiplication

Everything becomes element-wise multiplication in Fourier space:

$$\widehat{Pg_x}[k] = \hat{g}_x[k] - \text{DEL\_X}[k] \cdot \text{LAP\_INV}[k] \cdot \left(\text{DEL\_X}[k] \cdot \hat{g}_x[k] + \text{DEL\_Y}[k] \cdot \hat{g}_y[k]\right)$$

In code:
```python
div_g_hat = DEL_X * gx_hat + DEL_Y * gy_hat         # divergence of g
Pgx_hat   = gx_hat - DEL_X * LAP_INV * div_g_hat   # project x-component
Pgy_hat   = gy_hat - DEL_Y * LAP_INV * div_g_hat   # project y-component
```

No iteration required — just a handful of array multiplications.

---

## Part 10: The Pseudo-Spectral Method

### 10.1 The Problem with Products

A purely spectral method would compute *everything* in Fourier space. But the term $\theta \cdot \nabla\Delta^{-1}\theta$ involves a **product of two functions in physical space**.

In Fourier space, a product becomes a **convolution**:
$$\widehat{f \cdot g}[k] = \sum_{m} \hat{f}[m]\, \hat{g}[k-m]$$

Computing this naively costs $O(N^2)$ per wavenumber — far more expensive than FFTs.

### 10.2 The Pseudo-Spectral Solution

The **pseudo-spectral trick**: compute products in physical space (cheap, pointwise), and everything else in Fourier space:

1. Transform the fields you need from Fourier space to physical space (using IFFT)
2. Compute the product pointwise (simple array multiplication)
3. Transform the result back to Fourier space (using FFT)

Each FFT/IFFT costs $O(N^2 \log N)$ in 2D. The total cost is much lower than convolution.

### 10.3 The 5-Step RHS Pipeline

Every time the ODE solver requests $d\hat{\theta}/dt$, these 5 steps run:

**Step 1 — Compute $g = \theta\,\nabla\Delta^{-1}\theta$** (pseudo-spectral, in physical space)

```
theta_hat → LAP_INV * theta_hat → IFFT → psi   (stream-function-like field)
psi → DEL_X/DEL_Y → IFFT → ∂_x(psi), ∂_y(psi)   (gradients)
theta (from IFFT of theta_hat) × gradients → g_x, g_y   (nonlinear product)
g_x, g_y → FFT → gx_hat, gy_hat
```

**Step 2 — Apply Leray projection** (in Fourier space, pure multiplication)

```
(gx_hat, gy_hat) → Leray → (Pgx_hat, Pgy_hat)
```

**Step 3 — Apply inverse Laplacian** to get velocity $v = -\Delta^{-1}Pg$

```
vx_hat = -LAP_INV * Pgx_hat
vy_hat = -LAP_INV * Pgy_hat
```

**Step 4 — Normalise** to enforce enstrophy constraint $\|\nabla u\|_{L^2} = F$ (Parseval)

```
||∇v||_{L²} = (1/N²) * sqrt(sum of |DEL * v|²)
u_hat = F / ||∇v||_{L²} * v_hat
```

**Step 5 — Advect** to compute $d\hat{\theta}/dt = -\mathcal{F}[u \cdot \nabla \theta]$

```
DEL_X * theta_hat → IFFT → ∂_x(theta)
DEL_Y * theta_hat → IFFT → ∂_y(theta)
u_x * ∂_x(theta) + u_y * ∂_y(theta) → FFT → d(theta_hat)/dt
```

---

## Part 11: Time Integration with RK45

### 11.1 The ODE

After spectral discretisation, the PDE becomes a large system of ODEs:

$$\frac{d\hat{\theta}}{dt} = F(\hat{\theta}, t), \qquad \hat{\theta}(0) = \widehat{\theta_0}$$

where $F$ is the RHS from the 5 steps above and $\hat{\theta}$ has $N^2$ complex components.

### 11.2 What Is RK45?

The **Runge–Kutta 4/5** (Dormand–Prince) method is a classical adaptive ODE solver. Here is the idea:

- At each time step $t_n$, evaluate $F$ at several cleverly chosen intermediate points between $t_n$ and $t_n + h$
- Combine these evaluations to get a 4th-order and a 5th-order estimate of $\hat{\theta}(t_n + h)$
- Use the difference between these two estimates to measure the local error
- If the error is too large, halve the step size $h$ and retry; if it is very small, increase $h$ to speed up

This is called an **adaptive** time integrator — it automatically picks a step size that balances accuracy and speed.

**Why RK45 and not simpler methods?** Euler's method (first order) is too inaccurate. Fourth-order Runge–Kutta with a fixed step is accurate but wastes evaluations when the solution is smooth. RK45 is the standard workhorse for stiff-to-moderate ODEs and matches MATLAB's `ode45`.

In SciPy:
```python
sol = scipy.integrate.solve_ivp(
    rhs,
    t_span=(0, T),
    y0=theta_hat_flat_real,
    method='RK45',
    rtol=1e-6,
    atol=1e-8
)
```

### 11.3 Real/Imaginary Splitting

`solve_ivp` requires a real-valued state vector. The complex array $\hat{\theta} \in \mathbb{C}^{N^2}$ is stored as:

$$y = \left[\mathrm{Re}(\hat{\theta}_{\text{flat}}),\ \mathrm{Im}(\hat{\theta}_{\text{flat}})\right] \in \mathbb{R}^{2N^2}$$

At each RHS call, the code reconstructs $\hat{\theta}$ from $y$, performs the 5-step computation, and returns $d y/dt$ in real form.

---

## Part 12: The Resolution Check — When to Stop

### 12.1 Why the Simulation Must Stop

As the velocity stirs $\theta$ into finer and finer filaments, the scalar develops features at increasingly high wavenumbers. Eventually, these features become smaller than the grid spacing $dx = 1/N$ — the grid can no longer represent them. Beyond this point, the simulation is no longer physically meaningful.

### 12.2 What Is Conserved?

Under exact incompressible advection, **all $L^p$ norms are conserved**:

$$\|\theta(t)\|_{L^p} = \|\theta_0\|_{L^p} \quad \text{for all } t \geq 0, \text{ all } p \geq 1$$

This is a mathematical fact following from the change-of-variables formula for divergence-free flows. Numerically, these norms should remain constant as long as the grid can resolve the solution.

### 12.3 The Event Function

The code monitors:

$$e(t) = \max\!\left(\left|\|\theta\|_{L^2} - 1\right|,\ \left|\frac{\|\theta\|_{L^4}}{\|\theta_0\|_{L^4}} - 1\right|,\ \left|\frac{\|\theta\|_{L^8}}{\|\theta_0\|_{L^8}} - 1\right|\right) - \varepsilon, \qquad \varepsilon = 10^{-3}$$

When $e(t) > 0$ (any norm drifts by more than $0.1\%$), the solver stops.

**Why also use $L^4$ and $L^8$?** The $L^2$ norm is quadratic and relatively forgiving. Higher $L^p$ norms are more sensitive to large values and sharp gradients — exactly the kind of structures that appear when the grid is under-resolving. Using three norms gives an earlier and more reliable warning.

---

## Recommended Books and Resources

### Introductory — Start Here

**"A Student's Guide to Fourier Transforms" — J.F. James (Cambridge University Press, 3rd ed., 2011)**
The most accessible introduction to Fourier transforms. Written specifically for science students without a heavy maths background. Short chapters, physical intuition, minimal prerequisites.

**"The Fourier Transform and Its Applications" — Ronald Bracewell (McGraw-Hill, 3rd ed., 2000)**
A classic with excellent hand-drawn visualisations. Bracewell's "picket fence" picture of the DFT and his sampling theorem explanations are famous for being unusually clear.

**"Fourier Analysis: An Introduction" — Elias Stein & Rami Shakarchi (Princeton Lectures in Analysis, Vol. 1, 2003)**
More rigorous, but very clearly written. Good for understanding *why* the theorems hold, not just how to use them. Freely available at many university libraries.

### Spectral / Numerical Methods — Directly Relevant

**"Spectral Methods in MATLAB" — Lloyd N. Trefethen (SIAM, 2000)** — *free PDF online*
The perfect companion to this project. 40 short programs (easily translated from MATLAB to Python) explaining spectral methods for PDEs. Very readable, starts from scratch. This book will make Section 3 of the report completely transparent.

**"Numerical Methods for Engineers" — Chapra & Canale (McGraw-Hill, 7th ed., 2015)**
Covers FFT, ODE solvers (including RK45), and numerical differentiation at an introductory engineering level. Very practical and hands-on.

**"A Practical Guide to Pseudospectral Methods" — Bengt Fornberg (Cambridge University Press, 1996)**
Specifically about pseudo-spectral methods — exactly the approach used in the mixing simulation. More technical than Trefethen's book, but the most thorough treatment of the subject.

**"Numerical Methods for Fluid Dynamics" — Dale Durran (Springer, 2nd ed., 2010)**
Focused on atmospheric science applications but covers spectral methods, pseudo-spectral methods, and time integration at a graduate level. Bridges the gap between numerical methods and physical simulation.

### PDEs and the Physical Problem

**"Introduction to Partial Differential Equations" — Walter Strauss (Wiley, 2nd ed., 2008)**
Good introductory PDE text covering Fourier series and transforms from a mathematical perspective. Accessible to anyone with calculus.

**"An Introduction to Fluid Dynamics" — G.K. Batchelor (Cambridge University Press, 2000)**
The definitive fluid dynamics reference. Chapters 1–3 are accessible and explain divergence-free velocity fields, the incompressibility condition, and passive scalar transport.

### Online Resources

| Resource | What It Covers |
|---|---|
| 3Blue1Brown "But What Is the Fourier Transform?" (YouTube) | The best 20-minute visual intro to Fourier transforms |
| 3Blue1Brown "But What Is a Fourier Series?" (YouTube) | Visual intuition for decomposing functions into waves |
| NumPy FFT documentation (`numpy.org/doc/stable/reference/routines.fft`) | Exact conventions used in the code |
| Trefethen's ATAP (free PDF at `people.maths.ox.ac.uk/trefethen`) | Rigorous approximation theory, including spectral methods |
| SciPy `solve_ivp` documentation | Full details of the RK45 implementation |

---

## Summary Table

| Concept | What It Is | Code Equivalent |
|---|---|---|
| DFT | Decompose discrete function into frequencies | `np.fft.fft2` |
| FFT | Fast $O(N\log N)$ algorithm for DFT | `np.fft.fft2` |
| Wavenumber $k$ | Frequency index; how many oscillations per unit length | `k_eff` |
| Centred wavenumbers | Remap DFT indices to $\{-N/2,\ldots,N/2\}$ | `k - N*(k > N//2)` |
| Nyquist frequency | Maximum representable frequency $= N/2$ | Index `N//2` |
| Spectral derivative | Multiply $\hat{f}$ by $2\pi i k$ | `DEL_X`, `DEL_Y` |
| Inverse Laplacian | Multiply $\hat{f}$ by $-1/(4\pi^2\|k\|^2)$ | `LAP_INV` |
| Parseval identity | $\|f\|_{L^2} = \|\hat{f}\|_{\ell^2}/N^2$ (2D) | Norm computations |
| $H^{-1}$ mix norm | Weighted norm that decreases with mixing | `LAMBDA_INV` |
| Leray projection | Extracts divergence-free part of a vector field | Step 2 of RHS |
| Pseudo-spectral | Products in physical space, derivatives in Fourier space | RHS evaluation |
| RK45 | Adaptive-step 4/5-order Runge–Kutta ODE solver | `solve_ivp` |
| Resolution check | Stop when $L^p$ norms drift beyond tolerance | `make_res_check()` |

---

*This guide accompanies the notebook `00_dft_for_beginners.ipynb` in the `python_code/` folder, which provides interactive Python demonstrations of every concept described here.*
