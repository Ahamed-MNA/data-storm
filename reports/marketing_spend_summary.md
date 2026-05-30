# Marketing Spend Optimization Summary Report

## Executive Summary
This report summarizes the optimal allocation of the **LKR 5,000,000** promotional budget across the Western Province outlets for **January 2026**. 
The optimization uses a logarithmic spend-to-volume response curve to model diminishing marginal returns, capped by Stochastic Frontier Analysis (SFA) demand ceilings.

| Metric | Value |
| :--- | :--- |
| **Total Promotional Budget** | LKR 5,000,000.00 |
| **Total Allocated Spend** | LKR 4,999,999.98 (100.00%) |
| **Expected Total Volume Lift** | 1,238,686.33 Liters |
| **Average Return on Investment (ROI)** | 0.24774 Liters/LKR |
| **Active Outlets Targeted (> 0 LKR)** | 1,981 / 8,989 (22.0%) |
| **Outlets Pushed to SFA Demand Ceiling** | 2 |
| **Diminishing Return Scale Parameter ($b$)** | 0.0005 |

---

## Allocation by Distributor
This table shows how the budget is distributed across the distributors in the Western Province.

| Distributor ID | Number of Outlets | Active Outlets | Total Spend (LKR) | Share of Spend (%) | Volume Lift (Liters) | Avg ROI (L/LKR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| DIST_W_01 | 3,016 | 663 | LKR 1,704,083.88 | 34.08% | 424,565.72 L | 0.24915 |
| DIST_W_02 | 2,983 | 659 | LKR 1,608,822.03 | 32.18% | 393,288.83 L | 0.24446 |
| DIST_W_03 | 2,990 | 659 | LKR 1,687,094.07 | 33.74% | 420,831.78 L | 0.24944 |

---

## Allocation by Outlet Size
Diminishing return optimization tends to favor mature, larger-volume outlets while capping them at their frontier.

| Outlet Size | Number of Outlets | Active Outlets | Total Spend (LKR) | Share of Spend (%) | Volume Lift (Liters) | Avg ROI (L/LKR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Extra Large | 428 | 428 | LKR 2,669,145.58 | 53.38% | 766,169.79 L | 0.28705 |
| Large | 1,267 | 1,243 | LKR 1,846,116.82 | 36.92% | 373,357.47 L | 0.20224 |
| Medium | 2,608 | 112 | LKR 180,219.50 | 3.60% | 37,005.35 L | 0.20533 |
| Small | 4,590 | 193 | LKR 297,838.60 | 5.96% | 60,815.86 L | 0.20419 |
| Unknown | 96 | 5 | LKR 6,679.48 | 0.13% | 1,337.86 L | 0.20029 |

---

## Allocation by Outlet Type
Different retail formats exhibit different sales volumes and potential.

| Outlet Type | Number of Outlets | Active Outlets | Total Spend (LKR) | Share of Spend (%) | Volume Lift (Liters) | Avg ROI (L/LKR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Bakery | 1,395 | 293 | LKR 755,457.49 | 15.11% | 187,973.72 L | 0.24882 |
| Eatery | 1,263 | 256 | LKR 598,233.34 | 11.96% | 145,768.13 L | 0.24366 |
| Grocery | 1,428 | 320 | LKR 765,689.94 | 15.31% | 186,699.85 L | 0.24383 |
| Hotel | 1,251 | 281 | LKR 770,734.21 | 15.41% | 194,587.86 L | 0.25247 |
| Kiosk | 1,184 | 279 | LKR 721,637.14 | 14.43% | 180,016.65 L | 0.24946 |
| Pharmacy | 1,228 | 299 | LKR 720,413.36 | 14.41% | 177,230.66 L | 0.24601 |
| SMMT | 1,240 | 253 | LKR 667,834.50 | 13.36% | 166,409.46 L | 0.24918 |

---
*Report generated on completion of the Optimization Module.*
