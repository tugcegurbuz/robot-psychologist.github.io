---
layout: post
title: " Prover-Verifier Games Improve Legibility of LLM Output"
categories: paper_reviews
---

In this post, we're reviewing the paper *[Prover-Verifier Games Improve Legibility of LLM Output](https://arxiv.org/abs/2407.13692)* from OpenAI.

### What’s the problem they’re tackling?

Large Language Models (LLMs) like GPT can solve math problems, but:

- Their reasoning (i.e., the steps they show in chain-of-thought) isn’t always **easy to understand or verify**.
- Even if the final **answer is right**, the explanation might be messy, confusing, or misleading.

The authors call this property **legibility** — how easy it is for *humans or simple models* to check and follow the reasoning.

### Why is this important?

If LLMs are going to assist humans in complex reasoning (like math, law, or science), then:

- Their steps should not just be correct, but **transparent and verifiable**.
- Ideally, even *smaller models* (or humans with limited time) should be able to check their work.

### What do they propose?

They introduce a **Prover-Verifier Game**, inspired by earlier work (Anil et al., 2021):

1. **Verifier**: A **smaller model** trained to check whether a solution is likely correct.
2. **Helpful Prover**: Tries to generate **correct and legible** math solutions that convince a verifier.
3. **Sneaky Prover**: Tries to generate **wrong** answers that look convincing enough to **fool** the verifier.

This creates an **adversarial training loop**:

- Verifiers get better at spotting incorrect reasoning.
- Helpful provers get better at writing solutions that are **both correct and clearly verifiable**.

### What did they find?

- Over training:
    - The **helpful prover becomes more accurate**, and
    - The **verifier becomes more robust**, even against sneaky attacks.
- When humans are asked to check these solutions (under time pressure):
    - They get **better** at verifying helpful prover outputs.
    - They get **worse** at catching sneaky prover outputs (i.e., fooled like the verifier).

This means the system is learning to produce reasoning that's:

- **Easier to check** if it's correct.
- **Harder to fake** if it's wrong.

### Why does this matter?

- This **legibility training** improves not just correctness, but also **human-alignment**: it makes outputs easier for real people to trust and evaluate.
- It’s a scalable method — using **small models as verifiers** lets you train large models without always needing human supervision.


### Big Picture of Their Method

They want to make LLM-generated math solutions **more legible** — i.e., easier for small models (and humans) to **check**. They achieve this by using a **Prover-Verifier Game**, training:

- a **Helpful Prover** to produce correct, checkable solutions;
- a **Verifier** to judge correctness;
- a **Sneaky Prover** to produce incorrect, but deceptively convincing solutions.

### Dataset Setup

- **Main task**: Solving [grade-school math (GSM)](https://huggingface.co/datasets/openai/gsm8k) problems.
- Each datapoint: a problem $x$, and a known correct answer $y(x)$.
- A solution $z$: a chain-of-thought (CoT) output including a final answer.
- Correctness of a solution: $1\text{correct}(x,z)=1$ if the answer matches the ground truth (verified using SymPy).

### Key Components

#### 1. **Prover $(\pi)$**

- Generates solution $z$ given input $x$.

- Two **modes**:
    - **Helpful**: tries to be correct and convincing.
    - **Sneaky**: tries to be incorrect but still fool the verifier.

#### 2. **Verifier $(v)$**

- Small model that outputs a **“convincingness score”** $v(x,z)\in[0,1]$ for solution $z$.

They initialize the prover and the verifier with different sized models that have been lightly finetuned on a variety of math data from the pretaining distribution, including human written proofs and derivations.

### Objectives and Losses

#### Verifier Objective

Train to **distinguish correct from incorrect** solutions using a **ranking-based loss** (Bradley-Terry model):

- Given a pair $z_1,z_2$, it should score the correct one higher.

- Objective:

$$
\mathbb{E}_{x \sim D_V} \left[ 
\mathbb{E}_{z_1, z_2 \sim \pi^*} \log \sigma \left( 
(V(x, z_1) - V(x, z_2)) \cdot \mathrm{sign}(\mathbb{1}_{\mathrm{correct}}(z_1) - \mathbb{1}_{\mathrm{correct}}(z_2)) 
\right) 
+ \lambda \mathbb{E}_{z \sim \pi^*} V(x, z)^2 
\right]
$$

- $\lambda$: regularization term to prevent score drift.

- $ D_V$: verifier’s own training set (disjoint from the prover’s).

💡 **Making sense of the math:** The first term is the reward modeling objective that maximizes log likelihood of correct solutions being ranked higher than incorrect solutions under the **Bradley-Terry model**, similar to Ouyang et al. (2022). The second term is a regularizer that is meant to center $V$ at 0. Without this term, adding an arbitrary per-problem constant to $V$ would not change the objective.



<div style="background-color: #f0f0f0; padding: 16px; border-radius: 8px;">

<h4><strong>What's the Bradley-Terry Model?</strong></h4>

<p>The <strong>Bradley-Terry model</strong> is a <strong>pairwise comparison model</strong> used to <strong>rank items</strong> based on relative preferences or strengths.</p>

<h4>Basic Idea:</h4>

<p>Given two items (e.g., solutions $z_1$ and $z_2$), the model defines the probability that $z_1$ is better than $z_2$ as:</p>

$$
P(z_1 \succ z_2) = \frac{e^{s(z_1)}}{e^{s(z_1)} + e^{s(z_2)}}

$$

<p>Where: $s(z_i)$ is a <strong>score</strong> assigned to item $z_i$ (in this paper: the verifier’s output $V(x, z_i)$).</p>

  


<p>In the paper, they use this framework to train the <strong>verifier</strong> to score <strong>correct solutions higher than incorrect ones</strong>. The loss encourages the verifier to rank a correct solution above an incorrect one using the <strong>log-likelihood</strong> of the Bradley-Terry model:</p>

$$
\log \sigma \left( (V(x, z_1) - V(x, z_2)) \cdot \mathrm{sign}(\mathbf{1}_{\mathrm{correct}}(z_1) - \mathbf{1}_{\mathrm{correct}}(z_2)) \right)
$$

</div>



#### Prover Objective

Trained using **Reinforcement Learning (PPO)**  to maximize an expected reward R derived from:

- Whether the output is **aligned** with the role (correct if helpful, wrong if sneaky).
- How **convincing** the verifier thinks it is (verifier score).

For the prover reward $r$, they have three variants that prioritizes verifier score $v$ and role-alignment $q$ differently. But the main one is:

**Signed Relative Convincingness (SRC)**

Reward is explained as:

> We experimented on three variants of $R$. We call the first reward variant the Signed Relative Convincingness (SRC) reward. Given a problem $x$, let the role indicator $\mathbf{1}_{\text{helpful}}(\text{role})$ be 1 if the prover is assigned to be helpful and 0 otherwise. Suppose that, conditioned on a role and $x$, the prover generates a batch of solutions $\{ z_0, \ldots, z_b \} \sim \pi(\cdot \mid x, \text{role})$ with verifier scores $\{ V(x, z_0), \ldots, V(x, z_b) \}$.
>
> Since the scores are not normalized, we first batch-normalize the scores to have mean 0, i.e.,
>
> $$
> V'(x, z_i) = V(x, z_i) - \left( \sum_{j=1}^{b} V(x, z_j) \right) / b,
> $$
>
> and then set the reward as follows:
>
> $$
> R_{\text{SRC}}(z_i \mid x, \text{role}) =
> \begin{cases}
> (2\mathbf{1}_{\text{correct}}(x, z_i) - 1) \cdot (2\mathbf{1}_{\text{helpful}}(\text{role}) - 1) \cdot V'(x, z_i) & \text{if } V'(x, z_i) > 0 \\
> -1 & \text{otherwise}
> \end{cases}
> $$



This case satisfies the assumptions that any role-aligned solution has higher score than any non-aligned one, and for any two role-aligned solutions, the one with higher verifier score gets higher reward. However, the second case where any solution below average score is directly assigned a score of -1 deviates from the theoretical formulation. The rationale for this case is to prioritize convincingness over role-alignment


This encourages the model to **be more convincing** when it’s aligned, and **less so** otherwise.

### Training Loop: Checkability Training

Training alternates between:

1. **Verifier Training Phase**
    - Trained on mixture of helpful/sneaky solutions from previous provers.
    - Learns to score correct solutions higher than incorrect ones.

    <br>
    Here's how the iterative training procedure rolls out over the training rounds:

    > The prover $\pi_*$ used for sampling solutions for verifier training is a mixture of an initial base prover and previous round provers, each balanced to have equal number of correct and incorrect solutions. Each of the previous rounds has equal ratio in the mixture, which is tuned as a hyperparameter. In round 0, the solutions are from the base prover sampled via a few-shot prompt. In later rounds of training, we replace part of the solutions with those from the new provers. Hence we always use the same number of solutions per problem. The amount of verifier optimization is constant over the rounds.
    > 
    
    Reasons for the mixture:
    
    - **Exposure to multiple solution styles**: Solutions from different training stages (or provers) may differ in **structure, errors, and expression**. This diversity forces the verifier to generalize, rather than overfitting to one style.
    - **Adversarial coverage**: Older provers may produce **easier-to-spot flaws**, while newer sneaky provers generate **subtler errors**. Mixing ensures the verifier learns to reject both.
    - **Balanced training**: They balance **correct vs. incorrect** samples in the verifier’s training set to ensure the verifier doesn't become biased towards one class.
    
    <br>
    So, using a **mixture of provers** across rounds helps train a **more stable and robust verifier**
    
2. **Prover Training Phase**
    - Uses PPO to maximize expected reward based on verifier’s feedback.
    - Samples role (“helpful” or “sneaky”) randomly.

Each round uses a new **verifier trained from scratch**, but informed by more robust solution examples over time.

#### Game-Theoretic Framing

They model this as a **Stackelberg game**:

- The **verifier leads** (trained first).
- The **prover follows**, adapting to beat the current verifier.
- If equilibrium is reached, prover solutions are legible to that verifier (robust checkability).


<div style="background-color: #f0f0f0; padding: 16px; border-radius: 8px;">

<h4><strong>Stackelberg Game (Leader-Follower Game)</strong></h4>

<p>In game theory, a <strong>Stackelberg game</strong> is a <strong>sequential</strong> game between two players:</p>

<ul>
  <li>One player is the <strong>leader</strong>: they make a decision <strong>first</strong>.</li>
  <li>The other is the <strong>follower</strong>: they observe the leader’s move and then respond <strong>optimally</strong>.</li>
</ul>

<p>The <strong>leader anticipates</strong> the best response of the follower and chooses its own strategy accordingly.</p>

In this paper:

<ul>
  <li>The <strong>verifier is the leader</strong>: it is trained <strong>first</strong> in each round.</li>
  <li>The <strong>prover is the follower</strong>: it’s optimized <strong>after</strong> the verifier, using RL to produce outputs that either:
    <ul>
      <li><strong>Fool</strong> the verifier (if sneaky), or</li>
      <li><strong>Convince</strong> it (if helpful).</li>
    </ul>
  </li>
</ul>

<p>This setup matches a <strong>Stackelberg equilibrium</strong> assumption: the prover best-responds to the fixed strategy of the verifier, and the verifier anticipates this when optimizing.</p>

<p>Hence, the training loop emulates a <strong>verifier-leading Stackelberg game</strong>, and under theoretical conditions, such games can converge to a setting where <strong>only correct and verifiable solutions</strong> persist (i.e., checkable protocols).</p>

<p>A <strong>Stackelberg equilibrium</strong> $(v^{*}, \pi^{*})$ is a point where:</p>

<ol>
  <li>Given verifier strategy $v^{*}$, the prover $\pi^{*}$ <strong>can’t improve</strong> its utility by changing its behavior (i.e., it is a best response).</li>
  <li>Given that the prover will best-respond, the verifier $v^{*}$ <strong>chooses its own strategy accordingly</strong> to maximize its own utility.</li>
</ol>

<p>Formally:</p>

<ul>
  <li>The <strong>prover</strong> solves: $\displaystyle \pi^{*} = \arg\max_{\pi} R_{P}(v^{*}, \pi)$</li>
  <li>Then the <strong>verifier</strong> solves: $\displaystyle v^{*} = \arg\max_{v} R_{V}(v, \pi^{*}(v))$</li>
</ul>

<p>This is <strong>unlike Nash equilibrium</strong>, where both players act simultaneously.</p>

<h4>Why does equilibrium mean legibility?</h4>

<p>In theory, if the <strong>Stackelberg equilibrium is reached</strong>:</p>

<ul>
  <li>The <strong>verifier can detect all incorrect (sneaky) solutions</strong>, and</li>
  <li>The <strong>helpful prover must output only correct, checkable solutions</strong> (since anything less won’t pass the verifier).</li>
</ul>

<p>This implies:</p>

<ul>
  <li><strong>Soundness</strong>: Verifier rejects all sneaky solutions.</li>
  <li><strong>Completeness</strong>: Verifier accepts all helpful, correct solutions.</li>
</ul>

<p>These are the <strong>checkability conditions</strong> from the paper — the very definition of <strong>legibility</strong>.</p>

<h4>Can we guarantee convergence?</h4>

<p><strong>Theoretically:</strong></p>

<ul>
  <li><strong>Theorem 1</strong> (in the paper) proves: If the model class is expressive enough (all deterministic functions), then the existence of a verifier-leading Stackelberg equilibrium is <strong>both necessary and sufficient</strong> for satisfying checkability (completeness + soundness).</li>
  <li>In other words: <em>if</em> you can reach this equilibrium, it ensures only correct/verifiable solutions persist.</li>
</ul>

<p><strong>But in practice:</strong></p>

<ul>
  <li>The players (verifier and prover) are deep neural networks — <strong>non-convex</strong>, <strong>stochastic</strong>, and <strong>finite capacity</strong>.</li>
  <li>So you can’t <em>guarantee</em> convergence to the true equilibrium.</li>
  <li>Instead, the paper uses <strong>coordinate descent</strong> (train verifier → train prover → repeat) and finds that it <strong>empirically converges</strong> to a helpful dynamic where:
    <ul>
      <li>The helpful prover becomes more accurate and clearer.</li>
      <li>The verifier becomes harder to fool.</li>
      <li>The sneaky prover becomes more subtle (but less successful).</li>
    </ul>
  </li>
</ul>

<p>So it's an <strong>approximate local equilibrium</strong>, which is good enough to yield meaningful <strong>legibility gains</strong>.</p>

</div>

### Variants of Prover Reward (Key to Training Stability)

1. **SRC** (Best-performing)
    - Encourages role-aligned solutions that score well under the verifier.
    - Uses batch-normalized scores.
    <br><br>
2. **CGC: Correctness-Gated Convincingness**
    - If the prover is role-aligned, use verifier score; otherwise, assign a fixed penalty (e.g., -2)
    - Leads to worse training dynamics due to clearer separation of styles.
    <br><br>
    
    Why does it hurt training?
    
    - Too harsh for exploration:
        - If a solution is role-mismatched (e.g., a helpful prover makes a small mistake), it gets **harshly penalized** with a fixed reward.
        - This discourages gradual improvement and can suppress learning signals — especially early in training where most outputs are imperfect.
        <br><br>
    - Leads to mode collapse:
        - The helpful and sneaky provers may **diverge into distinct, easily separable styles** (e.g., long/verbose vs. terse/faulty).
        - Then the verifier could learn **stylistic cues** rather than genuine correctness signals.
        <br><br>
    - No gradient signal for low-verifier-score-but-correct outputs:
        - A correct solution that has a low verifier score still gets a **positive reward** in SRC, allowing learning.
        - But in CGC, it might be punished if not sufficiently convincing, slowing progress.
    
    <br><br>
    The paper shows empirically (Fig. 9) that CGC leads to **less robust verifiers** and **sneaky provers that are easier to detect**. SRC (Signed Relative Convincingness) turned out to be more effective for training both sneaky and helpful agents without collapsing.
    
3. **Iterative Goodharting**
    - Just optimize for the verifier score regardless of correctness (i.e., fool the verifier).
    - Used to simulate adversarial attacks and stress-test the verifier.

### Conclusion: Why This Paper Matters & What’s Next

The **Prover‑Verifier Game** showcased here is more than a clever training trick — it’s a **practical recipe for making LLM reasoning *auditable***. By pitting a *helpful* and a *sneaky* prover against a constantly‑improving verifier, the authors demonstrate a scalable path to:

1. **Better Answers *and* Better Explanations**  
   - Accuracy rises *without* sacrificing clarity.  
   - Chain‑of‑thought becomes easier for both small models *and* rushed humans to check.

2. **Alignment that Scales with Model Size**  
   - A modest verifier can police a much larger prover, reducing the need for costly human review.  
   - The game‑theoretic framing gives a principled lens on why the loop converges toward legibility.

3. **A New Benchmark for “Hard‑to‑Fake” Reasoning**  
   - Success is not just “getting the right answer,” but doing so in a way that *resists* deceptive shortcuts.  
   - This raises the bar for future work on interpretability and safety.

#### Limitations & Open Questions

- **Domain Breadth**: Results are on GSM‑style math. Will the same dynamics hold for proofs, coding, scientific arguments, or policy analysis?  
- **Verifier Power**: What happens when sneaky provers become *much* larger than the verifier? Can iterated scaling keep up?  
- **Human Factors**: Even helpful provers can write dense derivations. How do we quantify *human* legibility, not just model legibility?

#### Where Could This Go?

1. **Multi‑Step Debates**  
   Extend from single‑shot verification to back‑and‑forth argumentation like an “AI debate,” giving verifiers room to ask follow‑ups.

2. **Tool‑Augmented Verifiers**  
   Plug in symbolic math, theorem provers, or fact‑checking APIs to boost detection of subtle errors.

3. **Cross‑Task Generalization**  
   Train a single verifier that guards reasoning across math, code, and natural‑language claims — moving toward a *general auditor*.


### Take‑Home Message

If we want LLMs to **reason *with* us rather than *for* us**, we need outputs that are *convincingly correct* ***and*** *transparently verifiable*.  
The Prover‑Verifier Game is a promising step in that direction.
