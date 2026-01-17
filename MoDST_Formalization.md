# MoDST: Modality-Dropout Safety Tripwires

## 1. Method Formalization

Let a Vision-Language Model (VLM) be represented as a conditional probability distribution $P_{\theta}(y \mid I, T)$, where $I$ is an input image and $T$ is a textual prompt. MoDST introduces a lightweight, inference-time verification layer composed of three distinct passes.

### 1.1 Inference Passes
MoDST executes the following passes in parallel (or sequence):

1.  **Full Context Pass ($y_{full}$):** The standard model output.
    $$y_{full} = \text{argmax } P_{\theta}(y \mid I, T)$$
2.  **Text-Only Tripwire ($y_{text}$):** The model is prompted with $T$ but $I$ is "dropped" (replaced by a null/zero tensor or a generic placeholder). 
    $$y_{text} = \text{argmax } P_{\theta}(y \mid \emptyset, T)$$
    *Goal: Detect if the answer is hallucinated from LLM priors or triggered by text-side biases.*
3.  **Image-Only Tripwire ($y_{img}$):** The model is prompted with $I$ but $T$ is replaced with a generic grounding prompt $T_{null}$ (e.g., "Describe the main object.").
    $$y_{img} = \text{argmax } P_{\theta}(y \mid I, T_{null})$$
    *Goal: Isolate visual features to ensure $y_{full}$ is actually grounded in visual evidence.*

### 1.2 Tripwire Score ($S$)
The Tripwire Score $S$ measures the **grounding consistency** and **fusion safety**. We define $S$ using a semantic similarity function $\text{sim}(\cdot, \cdot)$ (e.g., BERTScore or a lightweight embedding distance):

$$S = 1 - \frac{\text{sim}(y_{full}, y_{img}) + \text{sim}(y_{full}, y_{text})}{2}$$

- **High $S$ (Inconsistency):** Indicates that $y_{full}$ contradicts either the isolated visual evidence or the textual constraints, or that the fusion logic has failed (e.g., multi-modal jailbreak).
- **Low $S$ (Consistency):** Indicates the model is reaching the same conclusion through multiple evidentiary paths.

### 1.3 Safety Action Policy
Given a safety threshold $\tau$:

- **IF $S < \tau$:** **Deploy.** The response is deemed grounded and consistent.
- **IF $S \geq \tau$:** **Abstain/Fallback.** 
    - *Policy A (Abstain):* "I'm sorry, I cannot confidently answer this based on the provided image."
    - *Policy B (Clarification):* "I see [elements from $y_{img}$], but your prompt asks about [elements from $T$]. Could you clarify?"

---

## 2. Threat Model

MoDST is designed to mitigate **Multi-modal Fusion Failures**.

### 2.1 Target Failures
- **Multi-modal Jailbreaks:** Attacks where text is safe, and image is safe, but their combination triggers harmful output.
- **Over-reliance on Priors:** When the VLM ignores a conflicting image and follows a biased text prompt (e.g., "A doctor in a kitchen" $\to$ "A chef in a kitchen").
- **Hallucinated Grounding:** Asserting facts not present in the image but plausible in the text distribution.

### 2.2 Out of Scope
- **Pure Textual Safety:** MoDST assumes a baseline text safety filter is already in place.
- **Pixel-level Adversarial Attacks:** MoDST does not detect imperceptible noise meant to flip classifiers, unless that noise creates a semantic mismatch between passes.

---

## 3. Minimal Experimental Setup (4-Page Paper)

### 3.1 Setup
- **Models:** 
    - **LLaVA-v1.5-7B:** Representative of efficient, open-source VLMs.
    - **InstructBLIP-7B:** To show MoDST generalizes across different fusion architectures (Q-Former vs. Projection).
- **Datasets:**
    - **SugarCrepe:** A benchmark for vision-language misalignments using hard negative captions.
    - **Shifted-MSCOCO:** A modified MS-COCO where we introduce "safety tripwire" prompts (e.g., asking for non-existent dangerous objects to test hallucination-led safety breaches).

### 3.2 Metrics
1.  **Tripwire AUC (T-AUC):** Area under the ROC curve for detecting misaligned/unsafe samples using $S$.
2.  **Safety Precision @ 90% Recall:** Precision of MoDST in catching failures while maintaining 90% of correct grounding.
3.  **Inference Overhead:** Ratio of MoDST latency to standard latency (target $< 3x$, optimized $< 1.5x$ via batching).
