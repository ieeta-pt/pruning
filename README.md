# Adaptive Neural Network Pruning

## Overview  
Neural networks often contain a very large number of parameters, many of which are redundant. **Pruning** reduces this redundancy by removing unnecessary parameters, resulting in smaller, faster, and more efficient models.  

This repository implements a novel pruning method for **feedforward neural networks**. Support for more complex architectures, such as CNNs and Transformers, is planned for future work.  

---

## Method Characteristics  
According to pruning taxonomy, the method is:  
- **Unstructured**  
- **Global**  
- **Differential**  
- Applied **during training**  
- **Magnitude-based**  
- Uses a **soft-mask** approach  

---

## Key Idea  
The method introduces two new trainable parameters into gradient descent:  

- **`r`** — the pruning ratio (percentage of weights removed).  
  - Learned automatically, removing the need for manual tuning or grid search.  
- **`τ`** — the temperature parameter, which controls the softness of the pruning mask.  

Both parameters adapt to the problem at hand during training.  

---

## Additional Hyperparameters  
To train `r` and `τ`, the method requires:  
- Learning rate for `r`  
- Learning rate for `τ`  
- Initial values for `r` and `τ`  
- Regularization scale factor `λ`  

---

## Implementation Notes 
Certain gradient computations for `r` and `τ` require approximations—for example, estimating the probability density function (PDF) of the weight distribution. To achieve this, kernel density estimation (KDE) is used, with the computation performed via the FFT algorithm. These operations are implemented in **`kde_fft.py`**, which handles these cases where exact computation is not optimal or feasible.  

---

## Detailed Description  

### Master's Thesis

**Author:** Guilherme Pereira  
**Program:** Mathematical Engineering  
**Institution:** Faculty of Sciences, University of Porto (FCUP)  

**Scientific Orientation:**  
- João Pedro Pedroso, Associate Professor, Faculty of Sciences, University of Porto  
- Sónia Gouveia, Assistant Researcher, University of Aveiro

For a full mathematical formulation and an in-depth explanation, see:  
[AGradientBasedApproachToAdaptiveNeuralNetworkPruning.pdf](./AGradientBasedApproachToAdaptiveNeuralNetworkPruning.pdf)
