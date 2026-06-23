"""UI explanation text for advanced dashboard sections."""

from __future__ import annotations


def get_advanced_explanations(section_key: str, model_key: str) -> tuple[str, str]:
    """Return (technical, intuition) text for an advanced section."""
    state_term = {
        "model1": "assets",
        "model2": "capital",
        "model3": "assets or labor-state choices",
    }.get(model_key, "state variables")
    shock_term = {
        "model1": "income",
        "model2": "TFP",
        "model3": "wage",
    }.get(model_key, "shock")

    texts = {
        "diagnostics": (
            f"These diagnostics evaluate the numerical quality of the solved dynamic program. "
            f"They summarize convergence of value function iteration, check whether the policy rules over {state_term} "
            f"look well-behaved, and report whether the solution is piling up on grid boundaries. "
            f"The optional grid-convergence run compares the solved objects across alternative discretizations to see "
            f"whether the policy and value arrays are stable rather than artifacts of one particular grid choice.",
            f"Think of this panel as a health check for the model solution. If the solver converges quickly, policies "
            f"look smooth, and grid changes barely move the answer, then the advanced results are more trustworthy. "
            f"If the policy sits on edges or moves a lot when the grid changes, the model may still be solving, but "
            f"the interpretation should be more cautious."
        ),
        "regime": (
            f"This block post-processes the simulated {shock_term} state sequence. It turns the existing two-state path "
            f"into transition frequencies, spell lengths, and conditional summaries of simulated variables by regime. "
            f"No new economics is introduced here; it is a descriptive layer built from the already simulated Markov chain.",
            f"This section tells you what living in a low or high {shock_term} regime actually feels like in the simulated "
            f"economy. Long spells or high persistence mean shocks are not just noisy one-period blips, while big differences "
            f"in conditional means show how much the economy behaves differently across regimes."
        ),
        "distribution": (
            "The distribution lab repeats the existing simulation many times across seeds and, optionally, across a few "
            "different initial states. It then summarizes the cross-run distribution of means, standard deviations, and "
            "final-period outcomes. This is still the same representative-agent model and the same optimal policy rules; "
            "the only difference is that you are looking across many simulated histories instead of one path.",
            "A single simulated path can look unusually smooth or unusually volatile just by luck. The distribution lab shows "
            "the range of outcomes the model can generate under the same policy rules, helping you separate typical behavior "
            "from one especially lucky or unlucky simulation."
        ),
        "scorecard": (
            "The scorecard compares model-implied simulated moments to a chosen target set, either illustrative targets, "
            "uploaded targets, or moments built from preset FRED data. The weighted loss aggregates those gaps into one fit "
            "summary, but the table is more informative than the single number because it shows exactly which moments the model "
            "matches and which ones it misses.",
            "This panel answers a simple question: does the model produce time-series behavior that looks remotely like the data "
            "you care about? A low loss suggests the model is at least moving in the right direction, while large gaps tell you "
            "which dimensions of reality the current parameterization struggles to capture."
        ),
        "phase": (
            f"These figures reorganize the solved objects into drift plots and surfaces over {state_term} and the two {shock_term} "
            f"states. They show how next-period choices, value levels, and consumption or labor rules vary across the state space. "
            f"The goal is not to add new math, but to present the already solved policy arrays in a way that highlights curvature, "
            f"crossings, and state dependence.",
            f"These visuals let you see the mechanics of adjustment. Where the drift crosses zero, the model is neither pushing the "
            f"economy up nor down in that region. Steeper surfaces indicate stronger responses, while flatter areas reveal regions "
            f"where the agent behaves more passively."
        ),
        "welfare": (
            "The welfare section reads the already solved value function as a measure of lifetime utility and, where useful, maps "
            "it into certainty-equivalent style summaries. The optional counterfactual compares the current solve with a second solve "
            "under one changed parameter, holding the interpretation explicitly at the model level rather than treating it as a generic score.",
            "Welfare is the model's bottom-line ranking of different states or policy environments. If one configuration produces higher "
            "value across relevant states, the model says households or planners prefer living in that world, not just that one variable "
            "looks better in isolation."
        ),
        "calibration": (
            "The calibration lab wraps the existing solve, simulation, and moment-comparison pipeline inside a small parameter search. "
            "For each candidate parameter set, the app resolves the model, simulates it, computes moments, and measures the distance from "
            "the selected targets. Evaluation caps are there to keep the exercise lightweight and demo-friendly rather than turning it into "
            "a full estimation platform.",
            "Calibration is the app's way of asking which parameter values make the model behave more like the chosen data moments. A better "
            "fit can make the model more persuasive for a particular benchmark, but it does not prove the model is literally true, so the "
            "results should be read as disciplined tuning rather than definitive empirical proof."
        ),
        "risk": (
            "This panel elicits a CRRA risk aversion coefficient (sigma) from a sequence of binary lottery choices. The estimator keeps a "
            "posterior distribution over sigma on a grid and updates it after each choice using a soft (logit) likelihood. The next question "
            "is chosen from a small candidate set (varying the gamble probability and sure amount) to reduce expected posterior uncertainty "
            "as quickly as possible.",
            "You will repeatedly pick between a sure payoff and a risky payoff. If you tend to reject gambles unless the sure amount is high, "
            "the inferred sigma rises (more risk aversion). If you often accept gambles even when the sure payoff is close to the expected value, "
            "the inferred sigma falls (less risk aversion). The output is a distribution, not a single number, to reflect remaining uncertainty."
        ),
    }
    return texts[section_key]
