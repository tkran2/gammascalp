# Own the project in an interview

Read the memo, run `python -m gammascalp hedge --rebalances 63`, and derive the
P&L of one hedging interval on paper (question 3) before presenting this. An
options interviewer will ask you to do that derivation at the board.

## A 45-second explanation

"I wanted to see what is left of the Black-Scholes replication argument once
you hedge at discrete times, pay costs, and use the wrong volatility. I wrote a
pricing library and a hedging simulator, and in each of five studies I stated
the classical prediction, simulated it, and compared. Most held closely. Two
things surprised me. The standard rule of thumb for discrete hedging error
understates risk away from the money, where my simulated risk came out about
20 percent higher, and I worked out the sum that fixes it. And one of my own tests failed, the one asserting that drift
does not matter to a delta hedger, which turned out to be a real second-order
effect that I then derived and confirmed."

## Questions you should be able to answer

**1. What are delta, gamma, vega, and theta?**

Delta: change in option value per $1 move in the stock; also the hedge ratio.
Gamma: how fast delta changes; the curvature. Vega: change in value per point
of volatility. Theta: change in value per day with nothing else moving.
For the at-the-money three-month call here (S = K = 100, 20% vol, 3% rate):
price 4.36, delta 0.55, gamma 0.040, vega 0.198 per vol point, theta −0.026
per day.

**2. Why does a delta-hedged option position make or lose money?**

The hedge removes the first-order exposure to the stock. What is left is
curvature against time decay. Long gamma earns `1/2 * gamma * dS^2` on every
move and pays theta every day. Under Black-Scholes, theta is exactly the
expected gamma gain at the implied volatility. So the position makes money
when the stock moves more than implied volatility said it would.

**3. Derive the P&L of one hedging interval.**

Hold one option, short delta shares, over a short time dt. Taylor-expand the
option: `dV = theta dt + delta dS + 1/2 gamma dS^2`. The stock leg earns
`-delta dS`. Ignoring interest: `P&L = theta dt + 1/2 gamma dS^2`. The
Black-Scholes equation with zero rates says `theta = -1/2 gamma S^2
sigma_i^2`. Substitute:

```text
P&L = 1/2 * gamma * S^2 * [ (dS/S)^2 - sigma_i^2 dt ]
```

Realized variance minus implied variance, scaled by dollar gamma. Everything
in this project is a consequence of that line.

**4. Why does the hedging error fall like one over the square root of N?**

Write `(dS/S)^2 = sigma^2 dt Z^2` with Z standard normal. The interval's error
is `1/2 gamma S^2 sigma^2 dt (Z^2 - 1)`. It has mean zero and a standard
deviation proportional to dt, because the variance of `Z^2` is 2. There are N
independent intervals of length T/N, so the total variance is N times
something proportional to `(T/N)^2`: it falls like 1/N, the standard deviation
like 1/sqrt(N). I measured a slope of −0.491.

**5. Where does the rule of thumb come from and why does it fail off the money?**

The exact leading-order variance is the sum over hedge dates of
`1/2 sigma^4 dt^2 E[(gamma S^2)^2]`. Derman and Kamal evaluate that expectation
approximately and express the result through today's vega:
`sqrt(pi/4) * vega * sigma / sqrt(N)`. At the money that is accurate (I got
99.7% at 1,008 hedges). Away from the money it is not, because an
out-of-the-money option has low gamma today but will have high gamma on the
paths where the stock travels to the strike, and the hedging error is made on
exactly those paths. Today's vega does not see them. At strikes 90 and 110,
with 1,008 hedges, simulated risk was 18% and 23% above the rule; my quadrature
of the leading-order sum was within 0.6% there and within 2.2% at daily
hedging.

**6. You think implied vol is too low. You buy the option. Which volatility do
you hedge with?**

It is a choice between certainty and convenience.
- Hedge with the **realized** volatility (if you knew it): the final P&L is
  `V(realized) - V(implied)` on every path. Guaranteed in total, but the
  mark-to-market swings along the way, and you need to know the true vol.
- Hedge with the **implied** volatility: each instant earns
  `1/2 gamma S^2 (sigma_r^2 - sigma_i^2)`, which is positive. You never lose,
  and the P&L accrues smoothly, but the total depends on how long the stock
  sits near the strike where gamma is large.

In my run, at the base-case drift, both averaged $1.99; the standard deviations
were $0.17 and $0.92. With a drift of -40% the implied-vol hedger's mean fell
to $1.82 while the realized-vol hedger's stayed at $1.99. The implied-vol
hedger lost on none of 400,000 paths.

**7. So a delta hedger really does not care about drift?**

In continuous time, no. With discrete hedges, slightly. Over an interval
`E[(dS/S)^2] = sigma^2 dt + mu^2 dt^2`, and replication at rate r absorbs the
r part, leaving an extra `(mu - r)^2 dt^2` of squared move the seller has to
pay gamma on. Summed up: `mean = -(mu - r)^2 (T/N) E[integral of 1/2 gamma
S^2]`. I found this because a test failed. At realistic drifts it is
invisible; at a drift of 100% a year with daily hedging it is 2% of premium.

Separately, when you hedge at the wrong volatility the mean also depends on
drift, for a different reason: drift decides how much time the stock spends
near the strike.

**8. What does Leland's adjustment do?**

With a proportional cost k per dollar traded, each rehedge costs about
`k * S * |change in delta|`, and the change in delta is `gamma * dS`. So costs
are proportional to gamma times the absolute move, which looks like extra
volatility for someone short gamma. Leland prices and hedges with
`sigma^2 * (1 + sqrt(8/pi) * k / (sigma * sqrt(dt)))`. In my run that brought
the seller's mean P&L back to zero within 0.1% of premium.

A subtlety I found in my own results: the markup equals the cost of the
*adjusted* hedge. Someone who charges the markup but keeps the plain delta
trades more and pays up to 8% more than the markup.

**9. Why hedge on a band instead of on a clock?**

A clock trades whether or not delta has moved. A band trades only when the
hedge is off by enough to matter. Whalley and Wilmott's asymptotic band has
half-width proportional to `(k * S * gamma^2 / risk aversion)^(1/3)`: wider
when costs are high, and wider where gamma is high because that is where a
tight band would trade constantly. At equal average cost it had 29% to 45%
less risk in my runs.

**10. Why can't you hedge jumps?**

Delta hedging works because over a short interval the option's change is
nearly linear in the stock's. A jump is a large move in zero time: the hedge is
at yesterday's delta when it lands, and the loss is the full convexity of the
option over the jump. No rehedging frequency helps. I matched total variance
between a stock with jumps and a smooth one: 63 times more hedging cut hedging
risk by 87% on the smooth stock and by 8% on the one with jumps (which still
has a diffusion part, so the jump component itself barely moved). You need other options to
hedge gamma across a gap.

**11. How do you compute implied volatility robustly?**

Price is increasing in volatility, so there is a unique root, but Newton's
method can overshoot where vega is tiny. I convert every option to its
out-of-the-money side by parity (better conditioning), start at the
Manaster-Koehler point where vega is largest, take Newton steps, and fall back
to bisection whenever a step leaves the bracket. I also refuse to answer when
the time value is below floating-point noise, since the answer there is
meaningless. One million options: about 780,000 per second, maximum error
3·10⁻¹³ where time value exceeds 10⁻⁴ of the forward.

**12. What is SVI and how do you know a fitted smile has no arbitrage?**

SVI writes total implied variance as a function of log-moneyness k with five
parameters: `w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))`. It is
linear in the wings, which is what Lee's moment formula says a smile must be
at most.

Butterfly arbitrage means the implied density is negative somewhere. Gatheral's
function `g(k)`, built from w and its first two derivatives, is proportional
to that density, so I check `g(k) >= 0` on a grid. Calendar arbitrage means
total variance decreases with maturity at some strike, so I check that slices
do not cross. I tested the butterfly check against a published arbitrageable
parameter set to make sure it can fail.

**13. Why fit SVI to Heston prices?**

To have a known truth. With real quotes I could not tell fitting error from
noise. Heston gives exact prices and a realistic skew. SVI fitted them to 2.9
basis points of volatility on average. With 10 bp of noise added to the
quotes, the fit was 5.4 bp from the true smile: closer than the quotes
themselves, because five parameters average the noise of 25 strikes.

**14. How did you validate the Heston pricer?**

Two different integral representations (Lewis's single integral and
Gil-Pelaez's pair) implemented separately agree to many digits. Setting
vol-of-vol to zero recovers Black-Scholes. Put-call parity holds. I also used
the formulation of the characteristic function that avoids the branch-cut
discontinuity ("the little Heston trap").

**15. Is any of this a trading strategy?**

No. Every path is simulated from a model I chose, and two of the studies hand
the hedger information nobody has. It tests whether I understand where option
P&L comes from and whether my code agrees with known mathematics.

**16. What would you do next?**

Hedge with a second option to control gamma and vega, which is the real answer
to the jump result. Run the hedging engine on historical prices, where the
realized-versus-implied question becomes an empirical one. Replace the
asymptotic band with a numerically solved optimal one.

## Numbers to know

| Fact | Value |
| --- | --- |
| Base option | 3-month ATM call, S = K = 100, vol 20%, rate 3%, price $4.3576 |
| Hedging runs simulated | 12,640,000 (10,240,000 distinct price paths) |
| Scaling exponent of hedging error | −0.491 ± 0.001 (theory −0.5) |
| Std of daily-hedged P&L | 10% of premium |
| Rule of thumb off the money (1,008 hedges) | simulated risk 18% to 23% higher; leading-order sum within 0.6% |
| Buy at 20%, realize 30% (base drift) | mean $1.99 both ways; std $0.17 vs $0.92 |
| Leland-adjusted mean | within 0.10% of premium |
| Band vs clock at equal cost | 29% to 45% less risk |
| 16 → 1,008 hedges | hedging risk −87% without jumps, −8% with |
| Tests | 58 |

## What not to claim

- Not a backtest, not a strategy, no real market data.
- The formulas are other people's. The second-order drift formula and the
  exact-sum correction are small derivations you should be able to reproduce,
  not research contributions.
- "Never lost on 400,000 paths" is a statement about a model with constant
  volatility that you knew in advance.
