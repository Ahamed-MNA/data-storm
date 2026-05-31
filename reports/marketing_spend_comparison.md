# Marketing Spend Optimization Comparison Report: Stage 1 vs. Stage 2

This report provides a comparative analysis of the marketing spend optimization outputs between **Stage 1** (initial data/predictions) and **Stage 2** (refined Silver/Gold feature engineering and re-fitted SFA prediction model).

---

## 1. Key Metrics Comparison

| Metric | Stage 1 | Stage 2 | Difference | % Change |
| :--- | :---: | :---: | :---: | :---: |
| **Total Promotional Budget** | LKR 5,000,000.00 | LKR 5,000,000.00 | LKR 0.00 | 0.00% |
| **Total Allocated Spend** | LKR 4,999,999.98 | LKR 4,999,999.97 | -LKR 0.01 | ~0.00% |
| **Expected Total Volume Lift** | 1,238,686.33 Liters | 1,238,686.46 Liters | +0.13 Liters | +0.00% |
| **Average Return on Investment (ROI)** | 0.24774 Liters/LKR | 0.24774 Liters/LKR | 0.00000 | 0.00% |
| **Active Outlets Targeted (> LKR 0)** | 1,981 / 8,989 | 1,981 / 8,989 | 0 | 0.00% |
| **Outlets Reaching SFA Demand Ceiling** | 2 | 0 | -2 | -100.00% |
| **Total Available Headroom** | 5,035,908.00 Liters | 7,836,759.53 Liters | +2,800,851.53 Liters | +55.62% |
| **Optimal Lagrange Multiplier ($\lambda$)** | ~0.151859 | 0.151855 | -0.000004 | -0.00% |

---

## 2. Mathematical Insight: Why Are the Results So Stable?

Despite the refined SFA modeling in Stage 2 causing a **55.62% increase in total available headroom** (expanding from ~5.04M to ~7.84M Liters), the optimal spend allocations and total lift changed by less than **0.01%**. 

This stability is driven by the underlying optimization formulation:
1. **Convex Formulation & Lagrange Multiplier:** The spend allocated to outlet $i$ is calculated as:
   $$x_i^*(\lambda) = \text{clip}\left(\frac{a_i}{\lambda} - \frac{1}{b}, 0, U_i\right)$$
   where $a_i = \max(Y_{\text{historical}, i}, 1.0)$ and $b = 0.0005$ are constants derived from historical transactions (which did not change between stages).
2. **Ceiling Constraints Are Inactive:** The budget of **LKR 5 Million** is extremely small compared to the total spend required to push all outlets to their SFA ceiling ($4.4 \times 10^{26}$ LKR due to the exponential ceiling bounds $U_i = \frac{\exp(H_i / a_i) - 1}{b}$). Thus, the ceilings $U_i$ are inactive for almost all outlets (only 2 hit them in Stage 1; 0 in Stage 2).
3. Since $a_i$ is identical and the ceilings $U_i$ are inactive, the optimal allocation $x_i^*$ is purely a function of the Lagrange multiplier $\lambda$, which solves to almost the exact same value to distribute the fixed LKR 5M budget.

---

## 3. Detailed Outlet-Level Analysis

A full comparison reveals that the **total absolute difference in spend across all 8,989 outlets is only LKR 287.43**. 

Only **2 outlets** experienced a spend change greater than LKR 1.00. These were precisely the two outlets whose SFA ceilings were active in Stage 1:

### OUT_00300
- **Historical January Sales:** 685.66 Liters
- **Stage 1 Spend Allocation (Capped):** LKR 2,386.28
- **Stage 2 Spend Allocation (Uncapped):** LKR 2,515.26
- **Difference:** **+LKR 128.98**
- **Rationale:** The new Stage 2 frontier prediction for this outlet increased to **1,400.80 Liters** (headroom of **715.14 Liters**), lifting the SFA ceiling and allowing the optimizer to allocate more budget.

### OUT_00130
- **Historical January Sales:** 668.32 Liters
- **Stage 1 Spend Allocation (Capped):** LKR 2,386.29
- **Stage 2 Spend Allocation (Uncapped):** LKR 2,401.02
- **Difference:** **+LKR 14.73**
- **Rationale:** The new Stage 2 frontier prediction increased to **1,365.36 Liters** (headroom of **697.04 Liters**), releasing the ceiling cap.

### Other Outlets
To fund the combined **+LKR 143.71** increase for these two outlets, the optimizer distributed minor reductions of **LKR 0.15 to LKR 0.16** across the other **1,979 active outlets**. Outlets that received LKR 0 in Stage 1 continue to receive LKR 0 in Stage 2.

---

## 4. Allocation Breakdown Stability

The high stability is also reflected in the distributor and outlet segment distributions:

### Share of Spend (%) by Distributor
- **DIST_W_01:** 34.08% (Stage 1) $\rightarrow$ 34.08% (Stage 2)
- **DIST_W_02:** 32.18% (Stage 1) $\rightarrow$ 32.18% (Stage 2)
- **DIST_W_03:** 33.74% (Stage 1) $\rightarrow$ 33.74% (Stage 2)

### Share of Spend (%) by Outlet Size
- **Extra Large:** 53.38% (Stage 1) $\rightarrow$ 53.38% (Stage 2)
- **Large:** 36.92% (Stage 1) $\rightarrow$ 36.92% (Stage 2)
- **Small:** 5.96% (Stage 1) $\rightarrow$ 5.96% (Stage 2)
- **Medium:** 3.60% (Stage 1) $\rightarrow$ 3.60% (Stage 2)
- **Unknown:** 0.13% (Stage 1) $\rightarrow$ 0.13% (Stage 2)

---

## 5. Conclusion

The Stage 2 model updates have confirmed that the marketing spend optimization is **exceptionally robust**. While the refined SFA modeling corrected prediction ceilings and significantly increased total potential headroom (+55.62%), the fixed promotional budget of LKR 5M is far below the constraint thresholds. The allocation continues to prioritize high-volume **Extra Large** and **Large** outlets where the marginal return (ROI) is maximized. 
The generated final allocation file is saved at [data_mavericks_budget_allocations.csv](file:///c:/data-storm/data_mavericks_budget_allocations.csv).
