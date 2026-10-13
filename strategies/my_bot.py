# Name: Parth Mahesh Deshmukh
# College: Thapar Institute of Engineering & Technology
# Roll Number: 1024031045

from __future__ import annotations

import math
import random
import statistics


class Bot:
    name = "OracleBot"

    # tuned by coordinate sweeps over 20 to 40 seeds per opponent, then
    # rechecked jointly against every reference and adversarial bot
    SHADE = 0.40
    TAU2 = 3.5
    TAKER_FORESIGHT_FRACTION = 1.0
    # measured forced fill rate, about 30 percent of rounds end at the midpoint
    FORCE_PROB_ESTIMATE = 0.32
    # quote at the round floor: the width premium costs more than the pick off
    # protection a wider quote buys
    WIDTH_EXTRA = 0
    # ticks of trading edge per tick of uncertainty a foresight leak removes
    FORESIGHT_TRADE_K = 0.15

    _N = statistics.NormalDist()

    def reset(self, seat, config, seed):
        self.seat = seat
        self.config = config
        self.rng = random.Random(seed)
        # round -> our cleaned estimate of their revealed sum, from their open quote
        self._reads = {}

    # pricing layer

    def _flat_threshold(self, r):
        n = self.config.REVEAL_PER_ROUND * r
        return math.sqrt(2 * n / math.pi)

    def _call_value(self, mu, sigma, strike):
        """E[max(0, X - strike)] for X normal with this mean and sd"""
        if sigma <= 0:
            return max(0.0, mu - strike)
        d = (strike - mu) / sigma
        return sigma * self._N.pdf(d) - (strike - mu) * (1 - self._N.cdf(d))

    def _record_read(self, obs, bid_p, ask_p):
        """Store what their opening quote says about their own revealed sum.

        An honest maker centres on their revealed sum, but if they held
        FORESIGHT that round their centre also contains a sample of OUR
        revealed coins, so that part has to come back out or we count our
        own coins twice.
        """
        r = obs.round
        n_r = self.config.REVEAL_PER_ROUND * r
        mid = (bid_p + ask_p) / 2.0
        leak = 0.0
        if "FORESIGHT" in obs.powers_theirs and n_r > 0:
            f = min(self.config.POWERS["FORESIGHT"]["magnitude"], n_r)
            leak = (f / n_r) * obs.k_mine
        self._reads[r] = mid - leak

    def _their_posterior(self, obs):
        """Mean and variance of their revealed sum, from foresight and past quotes."""
        r = obs.round
        n_r = self.config.REVEAL_PER_ROUND * r

        # our own foresight sample is exact, whatever it did not cover stays prior
        f_sum = sum(obs.foresight)
        unsampled = max(0, n_r - len(obs.foresight))
        mean_a, var_a = float(f_sum), float(unsampled)
        if var_a <= 1e-12:
            return mean_a, 0.0

        # their most recent opening quote, aged by the coins revealed since
        usable = [ra for ra in self._reads if ra <= r]
        if not usable:
            return mean_a, var_a
        ra = max(usable)
        mean_b = self._reads[ra]
        var_b = self.TAU2 + self.config.REVEAL_PER_ROUND * (r - ra)
        if var_b <= 1e-12:
            return mean_b, 0.0

        pa, pb = 1.0 / var_a, 1.0 / var_b
        return (mean_a * pa + mean_b * pb) / (pa + pb), 1.0 / (pa + pb)

    def _estimate(self, obs):
        their_mean, their_var = self._their_posterior(obs)
        seen = self.config.REVEAL_PER_ROUND * obs.round
        unrevealed = self.config.N_PRIVATE - seen
        mean_s = obs.k_mine + their_mean
        var_s = 2 * unrevealed + their_var
        return mean_s, var_s

    def _trade_edges(self, obs, bid_p, ask_p, mean_s, sigma):
        """Signed EV of buying at ask and of selling at bid, refunds included."""
        mu_buy = mean_s - ask_p
        mu_sell = bid_p - mean_s
        if "SUBSTITUTE" not in obs.powers_mine:
            return mu_buy, mu_sell
        cap = float(self.config.POWERS["SUBSTITUTE"]["magnitude"])
        # substitute refunds whatever the loss runs past the cap
        return (mu_buy + self._call_value(-mu_buy, sigma, cap),
                mu_sell + self._call_value(-mu_sell, sigma, cap))

    # bidding layer

    def _pickoff_cost(self, their_sd, width):
        """What a better informed taker takes off us at this opening width."""
        if their_sd <= 0:
            return 0.0
        return 2.0 * self._call_value(0.0, their_sd, width / 2.0)

    def _foresight_value(self, obs):
        """Three separate channels, two of them exactly derivable.

        The obligation pays the gap between our real straddle rate and the
        baseline rate it is scored against. A sharper centre also gets picked
        off less. The rest is plain trading edge from knowing S better.
        """
        r = obs.round
        n_r = self.config.REVEAL_PER_ROUND * r
        f = min(self.config.POWERS["FORESIGHT"]["magnitude"], n_r)
        width = min(obs.final_cap + self.WIDTH_EXTRA, obs.spread_cap)

        p_base = self.config.straddle_prob(r, width)
        sharp_unseen = self.config.N_COINS - n_r - f
        p_sharp = self.config.straddle_prob(r, width, unseen=sharp_unseen)
        oblig = self.config.MAKER_OBLIGATION * (p_sharp - p_base)

        _, their_var = self._their_posterior(obs)
        var_after = max(0.0, min(their_var, float(max(0, n_r - f))))
        sd_before, sd_after = math.sqrt(their_var), math.sqrt(var_after)
        pickoff = self._pickoff_cost(sd_before, width) - self._pickoff_cost(sd_after, width)
        trade = self.FORESIGHT_TRADE_K * (sd_before - sd_after)

        if obs.is_maker:
            return oblig + pickoff + trade
        return trade + self.TAKER_FORESIGHT_FRACTION * (oblig + pickoff)

    def _trick_room_value(self):
        return self.config.POWERS["TRICK_ROOM"]["magnitude"] * self.FORCE_PROB_ESTIMATE

    def _stealth_rock_value(self, obs):
        left = self.config.N_ROUNDS - obs.round + 1
        return (self.config.POWERS["STEALTH_ROCK"]["magnitude"]
                * self.FORCE_PROB_ESTIMATE * left)

    def _substitute_value(self, obs):
        _, var_s = self._estimate(obs)
        sigma = math.sqrt(max(var_s, 1e-9))
        cap = float(self.config.POWERS["SUBSTITUTE"]["magnitude"])
        return self._call_value(0.0, sigma, cap)

    def _transform_value(self, obs):
        """Option value of the swap, plus denial value if they look flat too."""
        n = self.config.REVEAL_PER_ROUND * obs.round
        sigma = math.sqrt(n)
        value = 2 * self._call_value(0.0, sigma, abs(obs.k_mine))
        earlier = [ra for ra in self._reads if ra < obs.round]
        if earlier:
            their_hat = self._reads[max(earlier)]
            if abs(their_hat) <= self._flat_threshold(max(earlier)):
                value += 2 * self._call_value(0.0, sigma, abs(their_hat))
        return value

    def _power_value(self, obs, name):
        if name == "FORESIGHT":
            return self._foresight_value(obs)
        if name == "TRICK_ROOM":
            return self._trick_room_value()
        if name == "SUBSTITUTE":
            return self._substitute_value(obs)
        if name == "STEALTH_ROCK":
            return self._stealth_rock_value(obs)
        if name == "TRANSFORM":
            return self._transform_value(obs)
        return 0.0

    def bid(self, obs, offered):
        if not offered or obs.te_mine <= 0:
            return {}
        salvage = self.config.TE_SALVAGE
        bids = {}
        for name in offered:
            v = self._power_value(obs, name)
            if v <= salvage:
                continue
            # round rather than truncate, and never let a worthwhile power go
            # uncontested: one TE only costs us salvage, which is 0.08 ticks
            te = max(1, int(v / salvage * self.SHADE + 0.5))
            bids[name] = min(te, obs.te_mine)
        if not bids:
            return {}
        # an over budget vector is zeroed by the engine, so fit it ourselves
        total = sum(bids.values())
        if total > obs.te_mine:
            scale = obs.te_mine / total
            bids = {n: int(v * scale) for n, v in bids.items()}
            bids = {n: v for n, v in bids.items() if v > 0}
        return bids

    # negotiation layer

    def quote(self, obs):
        mean_s, _ = self._estimate(obs)
        w = min(obs.final_cap + self.WIDTH_EXTRA, obs.spread_cap)
        lo = round(mean_s) - w // 2
        return (lo, lo + w)

    def _shift_total(self, powers):
        return sum(
            self.config.POWERS[n]["magnitude"]
            for n in ("TRICK_ROOM", "STEALTH_ROCK") if n in powers
        )

    def _final_turn_action(self, obs, bid_p, ask_p, mean_s, sigma):
        """Last turn as taker: accept, or counter to force the fill on our terms."""
        edge_buy, edge_sell = self._trade_edges(obs, bid_p, ask_p, mean_s, sigma)

        # countering makes us the last quoter, so we end up short at our own ask
        shift = self._shift_total(obs.powers_mine) - self._shift_total(obs.powers_theirs)
        force_price = ask_p + shift
        mu_force = force_price - mean_s
        force_edge = mu_force - self.config.FORCED_FILL_FEE
        if "SUBSTITUTE" in obs.powers_mine:
            cap = float(self.config.POWERS["SUBSTITUTE"]["magnitude"])
            force_edge += self._call_value(-mu_force, sigma, cap)

        best = max(edge_buy, edge_sell, force_edge)
        if best == force_edge:
            return ("COUNTER", ask_p, ask_p)
        if best == edge_buy:
            return "ACCEPT_BUY"
        return "ACCEPT_SELL"

    def respond(self, obs, quote, turn):
        bid_p, ask_p = quote
        if not obs.is_maker and obs.round not in self._reads:
            self._record_read(obs, bid_p, ask_p)

        mean_s, var_s = self._estimate(obs)
        sigma = math.sqrt(max(var_s, 1e-9))

        if turn == obs.n_turns:
            return self._final_turn_action(obs, bid_p, ask_p, mean_s, sigma)

        edge_buy, edge_sell = self._trade_edges(obs, bid_p, ask_p, mean_s, sigma)
        if edge_buy > 0 and edge_buy >= edge_sell:
            return "ACCEPT_BUY"
        if edge_sell > 0:
            return "ACCEPT_SELL"

        w = max(0, (ask_p - bid_p) - self.config.MIN_REDUCTION)
        centre = max(bid_p, min(round(mean_s), ask_p - w))
        return ("COUNTER", centre, centre + w)

    # transform layer

    def use_transform(self, obs):
        return abs(obs.k_mine) <= self._flat_threshold(obs.round)
