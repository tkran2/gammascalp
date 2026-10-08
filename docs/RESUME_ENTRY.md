# GammaScalp project entry

Use these bullets after you can derive the one-interval hedging P&L on paper
and explain why hedging at implied and at realized volatility give the same
mean but different risk. The project is ready to publish; no repository URL
exists until you push it (see `mac-and-github.md`). Do not describe it as a
backtest or a trading strategy: every path is simulated.

**GammaScalp | Options Hedging Research**
Python, NumPy, SciPy, Monte Carlo simulation, derivatives pricing

- Built an option-pricing and delta-hedging simulator (Black-Scholes, American
  trees, Heston, jump diffusion, SVI) and ran five studies testing classical
  results against 12.6 million simulated hedging runs; 58 automated tests.
- Showed the standard discrete-hedging risk formula understates risk away from
  the money (simulated risk 18% to 23% higher at 1,008 hedges) and derived a
  leading-order sum accurate to 0.6% there.
- Found delta-band hedging carried 29% to 45% less risk than fixed-interval
  hedging at equal cost; 63x more frequent hedging cut risk 87% for a pure
  diffusion but only 8% with jumps of equal total variance.

Shorter two-bullet version:

- Built an option-pricing and delta-hedging simulator and ran five studies
  against 12.6 million simulated hedging runs; found the standard
  discrete-hedging risk formula understates risk away from the money (simulated
  risk 18% to 23% higher) and derived a leading-order sum accurate to 0.6%.
- Showed delta-band hedging carried 29% to 45% less risk than fixed-interval
  hedging at equal cost, and that 63x more hedging cut risk 87% for a pure
  diffusion but only 8% with jumps of equal total variance.

Where it fits: first or second project on a quantitative trading resume, next
to QuoteLab (market making) and ZeroSum (game theory). On a quantitative
developer resume it can replace a web project if you want one research entry.

Each number above is in `results/manifest.json` or the tables of
`reports/GammaScalp_Research_Memo.md`. Know the qualifiers: a "hedging run" is
one hedger on one price path (10.2 million distinct paths); the 18% to 23% and
0.6% figures are at 1,008 hedges (at daily hedging the sum is within 2.2%); the
five studies are discrete hedging, volatility mismatch, trading costs, jumps,
and implied volatility with smile fitting.
