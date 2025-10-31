# Health Factors Analysis for Public Policy

## Overview
This research project examines the key determinants of life expectancy across global regions, focusing on three critical health factors: **fish and seafood consumption**, **Body Mass Index (BMI)**, and **cholesterol levels in blood**.  
The study provides evidence-based insights to inform **public health policy**, enabling governments to prioritize interventions that most effectively enhance population health and optimize healthcare expenditures.

## Objective
To quantify the impact of selected nutritional and physiological factors on **life expectancy**—used as the main indicator of population health—and identify which of these factors most strongly influence longevity.

## Variables Analyzed
| Category | Variable | Description | Role |
|-----------|-----------|-------------|------|
| **Dietary** | Fish and seafood consumption (kg/person/year) | Average per capita consumption of fish and seafood | Independent variable |
| **Physiological** | Body Mass Index (BMI) (male & female) | Weight-to-height ratio used as an indicator of body composition | Independent variable |
| **Physiological** | Cholesterol / Fat in blood (male & female) | Mean total cholesterol level per country (mmol/L) | Independent variable |
| **Health Indicator** | Life Expectancy (male & female) | Average lifespan at birth | Dependent variable |

## Methodology
- Collected and aggregated international health data (1980–2008) from eight world regions, covering approximately 200 countries, then sampled down to 40 for balanced regional representation.
- Computed **covariance** and **correlation coefficients** to assess the strength and direction of relationships between health factors and life expectancy.
- Constructed a **linear regression model** using `scikit-learn` to predict life expectancy based on fish consumption trends.
- Compared gender-based and regional variations to identify socioeconomic influences on health outcomes.
- Interpreted findings within the context of **public health economics** and **government policy planning**.

## Key Findings
- **Fish consumption** demonstrated a strong positive correlation with life expectancy in most regions (r ≈ 0.7–0.9).  
- **BMI** showed a moderate positive relationship with life expectancy in economically developed regions, reflecting higher healthcare standards and balanced nutrition.  
- **Cholesterol levels** were negatively associated with longevity in certain regions, consistent with cardiovascular health literature.  
- Regions with high average fish consumption, such as East Asia and Nordic countries, exhibited both higher life expectancy and more stable projected medical costs.

## Policy Implications
- Promoting fish consumption can serve as a **cost-effective public health strategy** to improve longevity and reduce national healthcare burdens.  
- Regions with low fish consumption, particularly in developing economies, could benefit from **nutritional subsidies** or **fish market support programs**.  
- Monitoring population-level BMI and cholesterol levels remains vital to maintaining sustainable public health outcomes.

## Tools and Technologies
- **Python**: `pandas`, `numpy`, `matplotlib`, `scikit-learn`  
- **Excel** for preprocessing and statistical summarization

## Supervision
Conducted under the supervision of **Dr. Seif El Dawlatly**,  
Department of Mathematics and Actuarial Science, Fundamentals of Data Science.
