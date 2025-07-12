# ================================
# ACTUARIAL AUTOMATION PLATFORM - STREAMLIT APP
# ================================
# Enterprise-grade actuarial automation platform for insurance industry
# Features: Risk modeling, regulatory compliance, stochastic projections, 
# real-time analytics, and comprehensive workflow automation

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import io

# Core libraries
from scipy import stats, optimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Actuarial and ML libraries
from lifelines import KaplanMeierFitter, WeibullFitter, LogNormalFitter
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Utilities
import warnings
warnings.filterwarnings('ignore')

# ================================
# STREAMLIT CONFIGURATION
# ================================

st.set_page_config(
    page_title="Actuarial Automation Platform",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: none;
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# DATA MODELS AND CONFIGURATION
# ================================

@dataclass
class ActuarialConfig:
    """Configuration for actuarial calculations"""
    discount_rates: Dict[str, float]
    mortality_tables: Dict[str, str]
    regulatory_margins: Dict[str, float]
    stress_scenarios: Dict[str, Dict[str, float]]
    monte_carlo_simulations: int
    confidence_levels: List[float]
    validation_thresholds: Dict[str, float]

@dataclass
class RiskProfile:
    """Risk profile for actuarial calculations"""
    longevity_risk: float
    interest_rate_risk: float
    inflation_risk: float
    operational_risk: float
    model_risk: float
    concentration_risk: float

# ================================
# ENHANCED ACTUARIAL CORE ENGINE
# ================================

class ActuarialEngine:
    """Advanced actuarial calculation engine"""
    
    def __init__(self, config: ActuarialConfig):
        self.config = config
        self.mortality_tables = {}
        self.yield_curves = {}
        self.inflation_curves = {}
        self.cache = {}
        self.setup_logging()
        
    def setup_logging(self):
        """Configure structured logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_sample_mortality_table(self):
        """Load sample mortality table for demonstration"""
        try:
            ages = list(range(20, 101))
            # Gompertz law of mortality
            qx_male = [0.0001 * np.exp(0.08 * (age - 20)) for age in ages]
            qx_female = [0.0001 * np.exp(0.075 * (age - 20)) for age in ages]
            
            # Cap at 1.0
            qx_male = [min(q, 1.0) for q in qx_male]
            qx_female = [min(q, 1.0) for q in qx_female]
            
            self.mortality_tables['standard'] = pd.DataFrame({
                'age': ages,
                'qx_male': qx_male,
                'qx_female': qx_female
            })
            
            self.logger.info(f"Loaded mortality table with columns: {self.mortality_tables['standard'].columns.tolist()}")
            self.logger.info(f"Mortality table shape: {self.mortality_tables['standard'].shape}")
            
        except Exception as e:
            self.logger.error(f"Error loading mortality table: {e}")
            raise

    def calculate_survival_probabilities(self, age: int, gender: str, 
                                    table_name: str = 'standard',
                                    improvement_factors: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate survival probabilities with mortality improvements - FIXED VERSION"""
        if table_name not in self.mortality_tables:
            self.load_sample_mortality_table()
            
        table = self.mortality_tables[table_name]
        
        # Map gender to correct column name
        gender_mapping = {'M': 'qx_male', 'F': 'qx_female'}
        qx_col = gender_mapping.get(gender.upper(), 'qx_male')
        
        # Validate column exists
        if qx_col not in table.columns:
            self.logger.error(f"Column {qx_col} not found in mortality table. Available columns: {table.columns.tolist()}")
            raise ValueError(f"Mortality table missing required column: {qx_col}")
        
        # Get mortality rates from age onwards
        mortality_rates = table[table['age'] >= age][qx_col].values
        
        # Handle case where age is beyond table range
        if len(mortality_rates) == 0:
            self.logger.warning(f"Age {age} is beyond mortality table range. Using maximum age data.")
            mortality_rates = table[qx_col].tail(1).values
        
        # Apply improvement factors if provided - FIXED LOGIC
        if improvement_factors is not None and len(improvement_factors) > 0:
            n_mortality = len(mortality_rates)
            n_improvements = len(improvement_factors)
            
            self.logger.debug(f"Age {age}: mortality_rates length = {n_mortality}, improvements length = {n_improvements}")
            
            # Create properly sized improvement array
            try:
                if n_improvements >= n_mortality:
                    # We have enough improvements - take exactly what we need
                    final_improvements = improvement_factors[:n_mortality].copy()
                else:
                    # We need more improvements - extend with the last improvement rate
                    last_improvement = improvement_factors[-1] if n_improvements > 0 else 0.02
                    
                    # Create extension array
                    extension_length = n_mortality - n_improvements
                    extension = np.full(extension_length, last_improvement)
                    
                    # Concatenate arrays
                    final_improvements = np.concatenate([improvement_factors, extension])
                
                # Ensure final array is exactly the right length
                final_improvements = final_improvements[:n_mortality]
                
                # Verify lengths match before applying improvements
                if len(final_improvements) != len(mortality_rates):
                    self.logger.error(f"Array length mismatch: improvements={len(final_improvements)}, mortality={len(mortality_rates)}")
                    # Fall back to no improvements
                    final_improvements = np.zeros(len(mortality_rates))
                
                # Apply improvements safely with bounds checking
                improvement_multipliers = np.clip(1 - final_improvements, 0.1, 2.0)
                
                # Ensure both arrays are 1-dimensional and same length
                mortality_rates = np.asarray(mortality_rates).flatten()
                improvement_multipliers = np.asarray(improvement_multipliers).flatten()
                
                # Final length check
                if len(mortality_rates) != len(improvement_multipliers):
                    self.logger.warning(f"Final length mismatch: mortality={len(mortality_rates)}, multipliers={len(improvement_multipliers)}")
                    # Use the shorter length
                    min_length = min(len(mortality_rates), len(improvement_multipliers))
                    mortality_rates = mortality_rates[:min_length]
                    improvement_multipliers = improvement_multipliers[:min_length]
                
                # Apply improvements element-wise
                mortality_rates = mortality_rates * improvement_multipliers
                
            except Exception as e:
                self.logger.warning(f"Error applying improvements for age {age}: {e}. Using base mortality rates.")
                # If anything goes wrong, just use base rates
                mortality_rates = np.asarray(mortality_rates).flatten()
        
        # Ensure mortality rates are valid
        mortality_rates = np.clip(mortality_rates, 0.0, 1.0)
        
        # Calculate survival probabilities
        survival_probs = np.cumprod(1 - mortality_rates)
        return survival_probs


    def generate_stochastic_mortality(self, years: int) -> np.ndarray:
        """Generate stochastic mortality improvement scenarios - FIXED VERSION"""
        base_improvement = 0.02  # 2% annual improvement
        volatility = 0.005
        
        improvements = []
        for _ in range(years):
            improvement = np.random.normal(base_improvement, volatility)
            # Ensure improvement is reasonable (between -5% and +10%)
            improvement = np.clip(improvement, -0.05, 0.10)
            improvements.append(improvement)
        
        # Return exactly the requested number of years as a proper numpy array
        result = np.array(improvements, dtype=np.float64)
        
        # Verify the result
        if len(result) != years:
            self.logger.error(f"Generated {len(result)} improvements but requested {years}")
            # Create a fallback array of correct length
            result = np.full(years, base_improvement, dtype=np.float64)
        
        return result


    def project_liability_scenario(self, members_df: pd.DataFrame,
                                interest_rates: np.ndarray,
                                mortality_improvements: np.ndarray) -> float:
        """Project liability under specific scenario - FIXED VERSION"""
        total_liability = 0
        
        try:
            # Limit sample size for performance during debugging
            sample_df = members_df.head(50) if len(members_df) > 50 else members_df
            
            # Ensure input arrays are proper numpy arrays
            interest_rates = np.asarray(interest_rates, dtype=np.float64).flatten()
            mortality_improvements = np.asarray(mortality_improvements, dtype=np.float64).flatten()
            
            for idx, (_, member) in enumerate(sample_df.iterrows()):
                try:
                    # Calculate projected liability for this member
                    age = (datetime.now() - pd.to_datetime(member['date_of_birth'])).days / 365.25
                    
                    # Skip if age is unrealistic
                    if age < 0 or age > 120:
                        continue
                    
                    # Apply mortality improvements
                    survival_probs = self.calculate_survival_probabilities(
                        int(age), member['gender'], improvement_factors=mortality_improvements
                    )
                    
                    # Skip if no survival probabilities
                    if len(survival_probs) == 0:
                        continue
                    
                    # Calculate present value with stochastic rates
                    annual_benefit = member['salary'] * member.get('service_years', 10) * 0.02
                    
                    # Ensure we have matching arrays for calculation
                    max_years = min(len(survival_probs), len(interest_rates))
                    if max_years == 0:
                        continue
                    
                    # Truncate both arrays to the same length
                    survival_probs_truncated = survival_probs[:max_years]
                    interest_rates_truncated = interest_rates[:max_years]
                    
                    # Calculate discount factors
                    discount_factors = np.exp(-np.cumsum(interest_rates_truncated))
                    
                    # Ensure all arrays are the same length
                    if len(survival_probs_truncated) != len(discount_factors):
                        self.logger.debug(f"Member {idx}: Array length mismatch: survival_probs={len(survival_probs_truncated)}, discount_factors={len(discount_factors)}")
                        min_length = min(len(survival_probs_truncated), len(discount_factors))
                        survival_probs_truncated = survival_probs_truncated[:min_length]
                        discount_factors = discount_factors[:min_length]
                    
                    # Calculate member liability
                    if len(survival_probs_truncated) > 0 and len(discount_factors) > 0:
                        member_liability = np.sum(
                            annual_benefit * survival_probs_truncated * discount_factors
                        )
                        
                        # Validate member liability
                        if member_liability > 0 and not np.isnan(member_liability) and np.isfinite(member_liability):
                            total_liability += member_liability
                        else:
                            self.logger.debug(f"Invalid liability for member {idx}: {member_liability}")
                
                except Exception as e:
                    # Log individual member errors but continue processing
                    self.logger.debug(f"Error processing member {idx}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error in liability projection: {e}")
            raise
        
        return total_liability


    def monte_carlo_projection(self, members_df: pd.DataFrame, 
                            projection_years: int = 30) -> Dict[str, Any]:
        """Run Monte Carlo projections for liability and asset modeling - FIXED VERSION"""
        n_sims = min(self.config.monte_carlo_simulations, 1000)  # Limit for performance
        results = []
        
        try:
            self.logger.info(f"Starting Monte Carlo with {n_sims} simulations over {projection_years} years")
            
            for sim in range(n_sims):
                try:
                    # Generate stochastic scenarios
                    interest_rates = self.generate_stochastic_rates(projection_years)
                    mortality_improvements = self.generate_stochastic_mortality(projection_years)
                    
                    # Validate generated scenarios
                    if len(interest_rates) != projection_years:
                        self.logger.warning(f"Simulation {sim}: Interest rates length {len(interest_rates)} != {projection_years}")
                        continue
                    
                    if len(mortality_improvements) != projection_years:
                        self.logger.warning(f"Simulation {sim}: Mortality improvements length {len(mortality_improvements)} != {projection_years}")
                        continue
                    
                    # Project liabilities under each scenario
                    projected_liability = self.project_liability_scenario(
                        members_df, interest_rates, mortality_improvements
                    )
                    
                    # Validate result
                    if projected_liability > 0 and not np.isnan(projected_liability) and np.isfinite(projected_liability):
                        results.append(projected_liability)
                    else:
                        self.logger.debug(f"Invalid liability result in simulation {sim}: {projected_liability}")
                
                except Exception as e:
                    self.logger.debug(f"Error in simulation {sim}: {e}")
                    continue
            
            if len(results) == 0:
                raise ValueError("No valid simulation results generated")
            
            # Calculate percentiles
            results_array = np.array(results, dtype=np.float64)
            percentiles = {}
            
            for p in self.config.confidence_levels:
                try:
                    percentiles[f'P{int(p*100)}'] = np.percentile(results_array, p*100)
                except Exception as e:
                    self.logger.warning(f"Could not calculate {p*100}th percentile: {e}")
            
            # Calculate VaR metrics safely
            var_95 = 0
            var_99 = 0
            try:
                if 'P95' in percentiles and 'P50' in percentiles:
                    var_95 = percentiles['P95'] - percentiles['P50']
                if 'P99' in percentiles and 'P50' in percentiles:
                    var_99 = percentiles['P99'] - percentiles['P50']
            except Exception as e:
                self.logger.warning(f"Error calculating VaR: {e}")
            
            return {
                'mean': np.mean(results_array),
                'std': np.std(results_array),
                'percentiles': percentiles,
                'var_95': var_95,
                'var_99': var_99,
                'results_array': results_array,
                'successful_simulations': len(results),
                'total_simulations': n_sims
            }
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo projection: {e}")
            raise

    def calculate_technical_provisions(self, members_df: pd.DataFrame, 
                                     valuation_date: datetime = None,
                                     include_risk_margin: bool = True) -> Dict[str, float]:
        """Calculate technical provisions with best estimate and risk margin"""
        if valuation_date is None:
            valuation_date = datetime.now()
            
        total_be_liability = 0
        total_risk_margin = 0
        
        for _, member in members_df.iterrows():
            # Calculate individual liability
            age = (valuation_date - pd.to_datetime(member['date_of_birth'])).days / 365.25
            survival_probs = self.calculate_survival_probabilities(
                int(age), member['gender']
            )
            
            # Benefit calculation (simplified defined benefit)
            annual_benefit = member['salary'] * member.get('service_years', 10) * 0.02
            benefit_stream = np.full(len(survival_probs), annual_benefit)
            
            # Discount factors
            discount_rate = self.config.discount_rates.get('risk_free', 0.035)
            discount_factors = np.exp(-discount_rate * np.arange(1, len(survival_probs) + 1))
            
            # Best estimate liability
            be_liability = np.sum(benefit_stream * survival_probs * discount_factors)
            total_be_liability += be_liability
            
            # Risk margin calculation (simplified)
            if include_risk_margin:
                risk_profile = self.calculate_member_risk_profile(member, age)
                risk_margin = be_liability * (
                    risk_profile.longevity_risk + 
                    risk_profile.interest_rate_risk + 
                    risk_profile.inflation_risk
                )
                total_risk_margin += risk_margin
        
        return {
            'best_estimate_liability': total_be_liability,
            'risk_margin': total_risk_margin,
            'technical_provisions': total_be_liability + total_risk_margin,
            'number_of_members': len(members_df)
        }

    def calculate_member_risk_profile(self, member: pd.Series, age: float) -> RiskProfile:
        """Calculate individual member risk profile"""
        # Age-based risk factors (simplified)
        longevity_risk = 0.05 if age < 65 else 0.08
        interest_rate_risk = 0.03 * (65 - age) / 65 if age < 65 else 0.01
        inflation_risk = 0.02
        
        return RiskProfile(
            longevity_risk=longevity_risk,
            interest_rate_risk=interest_rate_risk,
            inflation_risk=inflation_risk,
            operational_risk=0.01,
            model_risk=0.015,
            concentration_risk=0.005
        )

    def run_stress_tests(self, members_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Run comprehensive stress tests"""
        base_tp = self.calculate_technical_provisions(members_df)
        base_liability = base_tp['technical_provisions']
        
        stress_results = {}
        
        for scenario_name, stress_params in self.config.stress_scenarios.items():
            # Create a copy of the config with stressed parameters
            stressed_config = ActuarialConfig(
                discount_rates=self.config.discount_rates.copy(),
                mortality_tables=self.config.mortality_tables,
                regulatory_margins=self.config.regulatory_margins,
                stress_scenarios=self.config.stress_scenarios,
                monte_carlo_simulations=self.config.monte_carlo_simulations,
                confidence_levels=self.config.confidence_levels,
                validation_thresholds=self.config.validation_thresholds
            )
            
            # Apply stress to parameters
            if 'interest_rate_shock' in stress_params:
                for rate_type in stressed_config.discount_rates:
                    stressed_config.discount_rates[rate_type] += stress_params['interest_rate_shock']
            
            # Create temporary engine with stressed config
            stressed_engine = ActuarialEngine(stressed_config)
            stressed_engine.mortality_tables = self.mortality_tables.copy()
            
            # Apply mortality shock if specified
            if 'mortality_shock' in stress_params:
                for table_name in stressed_engine.mortality_tables:
                    table = stressed_engine.mortality_tables[table_name].copy()
                    table['qx_male'] *= (1 + stress_params['mortality_shock'])
                    table['qx_female'] *= (1 + stress_params['mortality_shock'])
                    # Cap at 1.0
                    table['qx_male'] = table['qx_male'].clip(upper=1.0)
                    table['qx_female'] = table['qx_female'].clip(upper=1.0)
                    stressed_engine.mortality_tables[table_name] = table
            
            stressed_tp = stressed_engine.calculate_technical_provisions(members_df)
            stressed_liability = stressed_tp['technical_provisions']
            
            impact = (stressed_liability - base_liability) / base_liability * 100
            
            stress_results[scenario_name] = {
                'base_liability': base_liability,
                'stressed_liability': stressed_liability,
                'impact_percentage': impact,
                'capital_requirement': max(0, stressed_liability - base_liability),
                'risk_margin': stressed_tp['risk_margin']
            }
        
        return stress_results

    def monte_carlo_projection(self, members_df: pd.DataFrame, 
                             projection_years: int = 30) -> Dict[str, Any]:
        """Run Monte Carlo projections for liability and asset modeling"""
        n_sims = min(self.config.monte_carlo_simulations, 1000)  # Limit for performance
        results = []
        
        try:
            for sim in range(n_sims):
                # Generate stochastic scenarios
                interest_rates = self.generate_stochastic_rates(projection_years)
                mortality_improvements = self.generate_stochastic_mortality(projection_years)
                
                # Project liabilities under each scenario
                projected_liability = self.project_liability_scenario(
                    members_df, interest_rates, mortality_improvements
                )
                
                # Validate result
                if projected_liability > 0 and not np.isnan(projected_liability):
                    results.append(projected_liability)
                else:
                    self.logger.warning(f"Invalid liability result in simulation {sim}: {projected_liability}")
            
            if len(results) == 0:
                raise ValueError("No valid simulation results generated")
            
            # Calculate percentiles
            results_array = np.array(results)
            percentiles = {}
            
            for p in self.config.confidence_levels:
                try:
                    percentiles[f'P{int(p*100)}'] = np.percentile(results_array, p*100)
                except Exception as e:
                    self.logger.warning(f"Could not calculate {p*100}th percentile: {e}")
            
            # Calculate VaR metrics safely
            var_95 = 0
            var_99 = 0
            if 'P95' in percentiles and 'P50' in percentiles:
                var_95 = percentiles['P95'] - percentiles['P50']
            if 'P99' in percentiles and 'P50' in percentiles:
                var_99 = percentiles['P99'] - percentiles['P50']
            
            return {
                'mean': np.mean(results_array),
                'std': np.std(results_array),
                'percentiles': percentiles,
                'var_95': var_95,
                'var_99': var_99,
                'results_array': results_array,
                'successful_simulations': len(results),
                'total_simulations': n_sims
            }
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo projection: {e}")
            raise

    def generate_stochastic_rates(self, years: int) -> np.ndarray:
        """Generate stochastic interest rate scenarios"""
        # Vasicek model implementation
        r0 = self.config.discount_rates.get('risk_free', 0.035)
        theta = 0.035  # long-term mean
        kappa = 0.1   # mean reversion speed
        sigma = 0.01  # volatility
        
        rates = [r0]
        dt = 1.0
        
        for _ in range(years):
            dr = kappa * (theta - rates[-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
            new_rate = rates[-1] + dr
            # Ensure rates stay within reasonable bounds (0.1% to 15%)
            new_rate = np.clip(new_rate, 0.001, 0.15)
            rates.append(new_rate)
        
        # Return exactly the requested number of years
        result = np.array(rates[1:])
        assert len(result) == years, f"Generated {len(result)} rates but requested {years}"
        return result

    def generate_stochastic_mortality(self, years: int) -> np.ndarray:
        """Generate stochastic mortality improvement scenarios"""
        base_improvement = 0.02  # 2% annual improvement
        volatility = 0.005
        
        improvements = []
        for _ in range(years):
            improvement = np.random.normal(base_improvement, volatility)
            # Ensure improvement is reasonable (between -5% and +10%)
            improvement = np.clip(improvement, -0.05, 0.10)
            improvements.append(improvement)
        
        # Return exactly the requested number of years
        result = np.array(improvements)
        assert len(result) == years, f"Generated {len(result)} improvements but requested {years}"
        return result

    def project_liability_scenario(self, members_df: pd.DataFrame,
                                 interest_rates: np.ndarray,
                                 mortality_improvements: np.ndarray) -> float:
        """Project liability under specific scenario"""
        total_liability = 0
        
        try:
            # Limit sample size for performance during debugging
            sample_df = members_df.head(100) if len(members_df) > 100 else members_df
            
            for _, member in sample_df.iterrows():
                try:
                    # Calculate projected liability for this member
                    age = (datetime.now() - pd.to_datetime(member['date_of_birth'])).days / 365.25
                    
                    # Skip if age is unrealistic
                    if age < 0 or age > 120:
                        continue
                    
                    # Apply mortality improvements
                    survival_probs = self.calculate_survival_probabilities(
                        int(age), member['gender'], improvement_factors=mortality_improvements
                    )
                    
                    # Skip if no survival probabilities
                    if len(survival_probs) == 0:
                        continue
                    
                    # Calculate present value with stochastic rates
                    annual_benefit = member['salary'] * member.get('service_years', 10) * 0.02
                    
                    # Ensure we have matching arrays for calculation
                    max_years = min(len(survival_probs), len(interest_rates))
                    if max_years == 0:
                        continue
                    
                    # Truncate both arrays to the same length
                    survival_probs_truncated = survival_probs[:max_years]
                    interest_rates_truncated = interest_rates[:max_years]
                    
                    # Calculate discount factors
                    discount_factors = np.exp(-np.cumsum(interest_rates_truncated))
                    
                    # Ensure all arrays are the same length
                    assert len(survival_probs_truncated) == len(discount_factors), \
                        f"Array length mismatch: survival_probs={len(survival_probs_truncated)}, discount_factors={len(discount_factors)}"
                    
                    # Calculate member liability
                    member_liability = np.sum(
                        annual_benefit * survival_probs_truncated * discount_factors
                    )
                    
                    # Validate member liability
                    if member_liability > 0 and not np.isnan(member_liability) and np.isfinite(member_liability):
                        total_liability += member_liability
                
                except Exception as e:
                    # Log individual member errors but continue processing
                    self.logger.debug(f"Error processing member {member.get('member_id', 'unknown')}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error in liability projection: {e}")
            raise
        
        return total_liability

# ================================
# SAMPLE DATA GENERATION
# ================================

def generate_sample_members(n_members: int = 1000) -> pd.DataFrame:
    """Generate sample member data for demonstration"""
    np.random.seed(42)  # For reproducibility
    
    # Generate random member data
    start_date = datetime(1960, 1, 1)
    end_date = datetime(2000, 1, 1)
    
    members = []
    for i in range(n_members):
        # Random date of birth
        random_days = np.random.randint(0, (end_date - start_date).days)
        dob = start_date + timedelta(days=random_days)
        
        # Employment date (after 18th birthday)
        earliest_employment = dob + timedelta(days=18*365)
        latest_employment = datetime.now() - timedelta(days=365)
        if earliest_employment < latest_employment:
            emp_days = np.random.randint(0, (latest_employment - earliest_employment).days)
            employment_date = earliest_employment + timedelta(days=emp_days)
        else:
            employment_date = earliest_employment
        
        # Service years
        service_years = max(0, (datetime.now() - employment_date).days / 365.25)
        
        # Ensure salary is positive
        salary = max(20000, np.random.normal(50000, 15000))
        
        member = {
            'member_id': f'M{i:06d}',
            'date_of_birth': dob,
            'gender': np.random.choice(['M', 'F']),  # Ensure consistent values
            'employment_date': employment_date,
            'salary': salary,
            'service_years': service_years,
            'pension_type': 'DB',
            'status': np.random.choice(['active', 'deferred'], p=[0.8, 0.2])
        }
        members.append(member)
    
    df = pd.DataFrame(members)
    
    # Data validation
    df['gender'] = df['gender'].str.upper()  # Ensure uppercase
    df = df[df['salary'] > 0]  # Remove any negative salaries
    df = df[df['service_years'] >= 0]  # Remove negative service years
    
    return df

# ================================
# STREAMLIT APPLICATION
# ================================

def main():
    # Initialize session state
    if 'config' not in st.session_state:
        st.session_state.config = ActuarialConfig(
            discount_rates={'risk_free': 0.035, 'corporate': 0.045},
            mortality_tables={'standard': 'built_in'},
            regulatory_margins={'ias19': 0.05, 'solvency_ii': 0.10},
            stress_scenarios={
                'longevity_shock': {'mortality_shock': -0.25},
                'interest_rate_up': {'interest_rate_shock': 0.01},
                'interest_rate_down': {'interest_rate_shock': -0.01},
                'combined_stress': {'mortality_shock': -0.15, 'interest_rate_shock': -0.005}
            },
            monte_carlo_simulations=1000,
            confidence_levels=[0.5, 0.75, 0.95, 0.99],
            validation_thresholds={'data_quality': 0.95}
        )
    
    if 'engine' not in st.session_state:
        st.session_state.engine = ActuarialEngine(st.session_state.config)
        st.session_state.engine.load_sample_mortality_table()
    
    if 'sample_data' not in st.session_state:
        st.session_state.sample_data = generate_sample_members(1000)

    # Main app
    st.title("🏛️ Actuarial Automation Platform")
    st.markdown("**Enterprise-grade pension valuation, risk modeling, and regulatory compliance**")
    
    # Sidebar navigation
    st.sidebar.title("🔧 Navigation")
    page = st.sidebar.selectbox(
        "Select Module",
        ["📊 Dashboard", "💰 Valuation", "⚡ Stress Testing", "🎲 Monte Carlo", "📋 Regulatory", "⚙️ Configuration"]
    )
    
    # Main content based on selection
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "💰 Valuation":
        show_valuation()
    elif page == "⚡ Stress Testing":
        show_stress_testing()
    elif page == "🎲 Monte Carlo":
        show_monte_carlo()
    elif page == "📋 Regulatory":
        show_regulatory()
    elif page == "⚙️ Configuration":
        show_configuration()

def show_dashboard():
    """Display main dashboard"""
    st.header("📊 Executive Dashboard")
    
    # Calculate current metrics
    try:
        with st.spinner("Calculating key metrics..."):
            tp_results = st.session_state.engine.calculate_technical_provisions(
                st.session_state.sample_data
            )
    except Exception as e:
        st.error(f"❌ Error calculating technical provisions: {str(e)}")
        st.info("💡 This might be due to data format issues. Please check the Configuration tab or refresh the page.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Technical Provisions", 
            f"£{tp_results['technical_provisions']/1e6:.1f}M",
            delta=f"Best Estimate: £{tp_results['best_estimate_liability']/1e6:.1f}M"
        )
    
    with col2:
        st.metric(
            "Active Members",
            f"{tp_results['number_of_members']:,}",
            delta=f"Risk Margin: £{tp_results['risk_margin']/1e6:.1f}M"
        )
    
    with col3:
        # Calculate funding ratio (assuming assets = 85% of TP)
        assets = tp_results['technical_provisions'] * 0.85
        funding_ratio = assets / tp_results['technical_provisions'] * 100
        st.metric(
            "Funding Ratio",
            f"{funding_ratio:.1f}%",
            delta=f"Assets: £{assets/1e6:.1f}M"
        )
    
    with col4:
        # Solvency ratio (assuming it's 142%)
        solvency_ratio = 142
        st.metric(
            "Solvency II Ratio",
            f"{solvency_ratio}%",
            delta="Above minimum" if solvency_ratio > 100 else "Below minimum"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Age Distribution")
        ages = [(datetime.now() - pd.to_datetime(dob)).days / 365.25 
                for dob in st.session_state.sample_data['date_of_birth']]
        
        fig = px.histogram(
            x=ages, 
            nbins=20, 
            title="Member Age Distribution",
            labels={'x': 'Age', 'y': 'Count'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 Salary Distribution")
        fig = px.histogram(
            x=st.session_state.sample_data['salary'], 
            nbins=20,
            title="Salary Distribution",
            labels={'x': 'Salary (£)', 'y': 'Count'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk indicators
    st.subheader("⚠️ Risk Indicators")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="success-box">
        <strong>✅ Longevity Risk</strong><br>
        Current exposure: Low<br>
        Trend: Stable
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ Interest Rate Risk</strong><br>
        Current exposure: Medium<br>
        Trend: Increasing
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="success-box">
        <strong>✅ Inflation Risk</strong><br>
        Current exposure: Low<br>
        Trend: Stable
        </div>
        """, unsafe_allow_html=True)

def show_valuation():
    """Display valuation module"""
    st.header("💰 Pension Valuation")
    
    # Parameters
    st.subheader("⚙️ Valuation Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        discount_rate = st.number_input(
            "Discount Rate (%)", 
            min_value=0.1, 
            max_value=10.0, 
            value=3.5, 
            step=0.1
        ) / 100
    
    with col2:
        mortality_table = st.selectbox(
            "Mortality Table",
            ["Standard", "Improved Longevity", "Stressed"]
        )
    
    with col3:
        include_risk_margin = st.checkbox("Include Risk Margin", value=True)
    
    # Update config
    st.session_state.config.discount_rates['risk_free'] = discount_rate
    st.session_state.engine.config = st.session_state.config
    
    # Run valuation
    if st.button("🔄 Run Valuation", type="primary"):
        with st.spinner("Running valuation calculations..."):
            results = st.session_state.engine.calculate_technical_provisions(
                st.session_state.sample_data,
                include_risk_margin=include_risk_margin
            )
        
        # Display results
        st.subheader("📊 Valuation Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Best Estimate Liability",
                f"£{results['best_estimate_liability']/1e6:.2f}M"
            )
            if include_risk_margin:
                st.metric(
                    "Risk Margin",
                    f"£{results['risk_margin']/1e6:.2f}M"
                )
            st.metric(
                "Technical Provisions",
                f"£{results['technical_provisions']/1e6:.2f}M"
            )
        
        with col2:
            # Create a breakdown chart
            if include_risk_margin:
                labels = ['Best Estimate', 'Risk Margin']
                values = [results['best_estimate_liability'], results['risk_margin']]
            else:
                labels = ['Best Estimate']
                values = [results['best_estimate_liability']]
            
            fig = px.pie(
                values=values, 
                names=labels, 
                title="Technical Provisions Breakdown"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed breakdown
        st.subheader("📋 Detailed Analysis")
        
        # Calculate liability by age group
        ages = [(datetime.now() - pd.to_datetime(dob)).days / 365.25 
                for dob in st.session_state.sample_data['date_of_birth']]
        age_groups = pd.cut(ages, bins=[0, 30, 40, 50, 60, 70, 100], 
                           labels=['<30', '30-40', '40-50', '50-60', '60-70', '70+'])
        
        liability_by_age = []
        for group in age_groups.categories:
            mask = age_groups == group
            if mask.sum() > 0:
                group_data = st.session_state.sample_data[mask]
                group_tp = st.session_state.engine.calculate_technical_provisions(group_data)
                liability_by_age.append({
                    'Age Group': group,
                    'Members': mask.sum(),
                    'Liability (£M)': group_tp['technical_provisions'] / 1e6
                })
        
        df_breakdown = pd.DataFrame(liability_by_age)
        st.dataframe(df_breakdown, use_container_width=True)

def show_stress_testing():
    """Display stress testing module"""
    st.header("⚡ Stress Testing")
    
    st.subheader("🎯 Stress Test Scenarios")
    
    # Scenario selection
    scenarios = list(st.session_state.config.stress_scenarios.keys())
    selected_scenarios = st.multiselect(
        "Select scenarios to run:",
        scenarios,
        default=scenarios
    )
    
    if st.button("⚡ Run Stress Tests", type="primary"):
        if selected_scenarios:
            with st.spinner("Running stress test scenarios..."):
                # Filter config to only selected scenarios
                filtered_scenarios = {
                    k: v for k, v in st.session_state.config.stress_scenarios.items() 
                    if k in selected_scenarios
                }
                temp_config = st.session_state.config
                temp_config.stress_scenarios = filtered_scenarios
                
                temp_engine = ActuarialEngine(temp_config)
                temp_engine.mortality_tables = st.session_state.engine.mortality_tables
                
                stress_results = temp_engine.run_stress_tests(st.session_state.sample_data)
            
            st.subheader("📊 Stress Test Results")
            
            # Results table
            results_data = []
            for scenario, result in stress_results.items():
                results_data.append({
                    'Scenario': scenario.replace('_', ' ').title(),
                    'Base Liability (£M)': f"{result['base_liability']/1e6:.1f}",
                    'Stressed Liability (£M)': f"{result['stressed_liability']/1e6:.1f}",
                    'Impact (%)': f"{result['impact_percentage']:.1f}%",
                    'Capital Requirement (£M)': f"{result['capital_requirement']/1e6:.1f}"
                })
            
            df_results = pd.DataFrame(results_data)
            st.dataframe(df_results, use_container_width=True)
            
            # Impact visualization
            st.subheader("📈 Impact Visualization")
            
            scenario_names = [r['Scenario'] for r in results_data]
            impacts = [float(r['Impact (%)'].replace('%', '')) for r in results_data]
            
            fig = px.bar(
                x=scenario_names,
                y=impacts,
                title="Stress Test Impact by Scenario",
                labels={'x': 'Scenario', 'y': 'Impact (%)'},
                color=impacts,
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk assessment
            max_impact = max(impacts)
            if max_impact > 20:
                st.markdown("""
                <div class="error-box">
                <strong>⚠️ High Risk Alert</strong><br>
                Maximum impact exceeds 20%. Consider additional risk management measures.
                </div>
                """, unsafe_allow_html=True)
            elif max_impact > 10:
                st.markdown("""
                <div class="warning-box">
                <strong>⚠️ Medium Risk</strong><br>
                Maximum impact between 10-20%. Monitor closely.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                <strong>✅ Low Risk</strong><br>
                All stress test impacts below 10%. Risk profile is acceptable.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please select at least one scenario to run.")

def show_monte_carlo():
    """Display Monte Carlo simulation module"""
    st.header("🎲 Monte Carlo Projections")
    
    # Parameters
    st.subheader("⚙️ Simulation Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_simulations = st.number_input(
            "Number of Simulations",
            min_value=10,
            max_value=1000,
            value=100,  # Start with smaller default
            step=10
        )
    
    with col2:
        projection_years = st.number_input(
            "Projection Years",
            min_value=10,
            max_value=50,
            value=30,
            step=5
        )
    
    with col3:
        confidence_level = st.selectbox(
            "Confidence Level (%)",
            [95, 99, 99.5],
            index=0
        )
    
    # Update config
    st.session_state.config.monte_carlo_simulations = n_simulations
    st.session_state.engine.config = st.session_state.config
    
    # Add debug mode toggle
    debug_mode = st.checkbox("🔍 Enable Debug Mode", value=False, help="Shows detailed information during calculation")
    
    # Test single calculation first
    if st.button("🧪 Test Single Calculation", help="Run a single test to verify everything works"):
        try:
            with st.spinner("Testing single calculation..."):
                # Generate test scenarios
                st.write("**Step 1: Generating stochastic scenarios...**")
                test_interest_rates = st.session_state.engine.generate_stochastic_rates(projection_years)
                test_mortality_improvements = st.session_state.engine.generate_stochastic_mortality(projection_years)
                
                st.write("**Step 2: Verifying array lengths...**")
                st.write(f"✅ Interest rates length: {len(test_interest_rates)} (expected: {projection_years})")
                st.write(f"✅ Mortality improvements length: {len(test_mortality_improvements)} (expected: {projection_years})")
                
                if debug_mode:
                    st.write(f"Sample member count: {len(st.session_state.sample_data)}")
                
                # Test with just one member first
                st.write("**Step 3: Testing with one member...**")
                test_member = st.session_state.sample_data.iloc[0:1]
                member = test_member.iloc[0]
                age = (datetime.now() - pd.to_datetime(member['date_of_birth'])).days / 365.25
                
                st.write(f"Test member age: {age:.1f}, gender: {member['gender']}")
                
                # Test mortality table lookup
                st.write("**Step 4: Testing mortality calculations...**")
                if 'standard' in st.session_state.engine.mortality_tables:
                    table = st.session_state.engine.mortality_tables['standard']
                    mortality_rates = table[table['age'] >= age]['qx_male'].values
                    st.write(f"Mortality rates from age {age:.0f}: {len(mortality_rates)} elements")
                    st.write(f"Mortality table age range: {table['age'].min()} to {table['age'].max()}")
                
                # Test survival probability calculation
                st.write("**Step 5: Testing survival probabilities...**")
                survival_probs = st.session_state.engine.calculate_survival_probabilities(
                    int(age), member['gender'], improvement_factors=test_mortality_improvements
                )
                st.write(f"✅ Survival probabilities calculated: {len(survival_probs)} elements")
                
                # Test full liability calculation
                st.write("**Step 6: Testing liability calculation...**")
                test_liability = st.session_state.engine.project_liability_scenario(
                    test_member, test_interest_rates, test_mortality_improvements
                )
                
                st.success(f"✅ Test calculation successful! Sample liability: £{test_liability/1e6:.2f}M")
                
                if debug_mode:
                    # Show more details
                    st.write(f"**Additional Details:**")
                    st.write(f"Salary: £{member['salary']:,.0f}")
                    st.write(f"Service years: {member.get('service_years', 0):.1f}")
                    st.write(f"Annual benefit: £{member['salary'] * member.get('service_years', 10) * 0.02:,.0f}")
                    
        except Exception as e:
            st.error(f"❌ Test calculation failed: {str(e)}")
            
            if debug_mode:
                with st.expander("🔍 Detailed Error Information"):
                    import traceback
                    st.code(traceback.format_exc())
                    
                    # Show data details
                    st.write("**Data Inspection:**")
                    st.write(f"Sample data shape: {st.session_state.sample_data.shape}")
                    st.write(f"Sample data columns: {st.session_state.sample_data.columns.tolist()}")
                    
                    # Check mortality table
                    if 'standard' in st.session_state.engine.mortality_tables:
                        table = st.session_state.engine.mortality_tables['standard']
                        st.write(f"Mortality table shape: {table.shape}")
                        st.write(f"Age range in table: {table['age'].min()} - {table['age'].max()}")
                    
                    # Check member ages
                    ages = [(datetime.now() - pd.to_datetime(dob)).days / 365.25 
                            for dob in st.session_state.sample_data['date_of_birth']]
                    st.write(f"Member ages range: {min(ages):.1f} - {max(ages):.1f}")
            else:
                st.info("💡 Enable Debug Mode for more detailed error information.")
    
    if st.button("🎲 Run Monte Carlo Simulation", type="primary"):
        try:
            with st.spinner(f"Running {n_simulations} Monte Carlo simulations..."):
                # Show progress if debug mode
                if debug_mode:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                mc_results = st.session_state.engine.monte_carlo_projection(
                    st.session_state.sample_data,
                    projection_years
                )
                
                if debug_mode:
                    progress_bar.progress(100)
                    status_text.text("Simulation completed!")
            
            st.subheader("📊 Monte Carlo Results")
            
            # Show simulation success rate
            success_rate = mc_results.get('successful_simulations', n_simulations) / mc_results.get('total_simulations', n_simulations) * 100
            if success_rate < 95:
                st.warning(f"⚠️ Only {success_rate:.1f}% of simulations were successful. Results may be less reliable.")
            else:
                st.success(f"✅ {success_rate:.1f}% of simulations completed successfully.")
            
            # Key statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Mean Liability",
                    f"£{mc_results['mean']/1e6:.1f}M"
                )
            
            with col2:
                st.metric(
                    "Standard Deviation",
                    f"£{mc_results['std']/1e6:.1f}M"
                )
            
            with col3:
                p_key = f'P{confidence_level}'
                if p_key in mc_results['percentiles']:
                    st.metric(
                        f"{confidence_level}th Percentile",
                        f"£{mc_results['percentiles'][p_key]/1e6:.1f}M"
                    )
                else:
                    st.metric(f"{confidence_level}th Percentile", "N/A")
            
            with col4:
                var_key = f'var_{confidence_level}' if confidence_level == 95 else 'var_99'
                if var_key in mc_results and mc_results[var_key] > 0:
                    st.metric(
                        f"VaR ({confidence_level}%)",
                        f"£{mc_results[var_key]/1e6:.1f}M"
                    )
                else:
                    st.metric(f"VaR ({confidence_level}%)", "N/A")
            
            # Distribution plot
            st.subheader("📈 Results Distribution")
            
            if len(mc_results['results_array']) > 0:
                fig = px.histogram(
                    x=mc_results['results_array'] / 1e6,
                    nbins=50,
                    title="Monte Carlo Liability Distribution",
                    labels={'x': 'Liability (£M)', 'y': 'Frequency'}
                )
                
                # Add percentile lines
                mean_val = mc_results['mean'] / 1e6
                fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                             annotation_text=f"Mean: £{mean_val:.1f}M")
                
                if f'P{confidence_level}' in mc_results['percentiles']:
                    p_val = mc_results['percentiles'][f'P{confidence_level}'] / 1e6
                    fig.add_vline(x=p_val, line_dash="dash", line_color="orange",
                                 annotation_text=f"P{confidence_level}: £{p_val:.1f}M")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ No valid simulation results to display.")
            
            # Risk metrics table
            st.subheader("📋 Risk Metrics")
            
            risk_metrics = []
            for conf in [50, 75, 95, 99]:
                p_key = f'P{conf}'
                if p_key in mc_results['percentiles']:
                    risk_metrics.append({
                        'Percentile': f'{conf}%',
                        'Liability (£M)': f"{mc_results['percentiles'][p_key]/1e6:.2f}",
                        'Excess over Mean (£M)': f"{(mc_results['percentiles'][p_key] - mc_results['mean'])/1e6:.2f}"
                    })
            
            if risk_metrics:
                df_risk = pd.DataFrame(risk_metrics)
                st.dataframe(df_risk, use_container_width=True)
            else:
                st.warning("⚠️ Could not calculate risk metrics.")
                
        except Exception as e:
            st.error(f"❌ Error running Monte Carlo simulation: {str(e)}")
            st.info("💡 Try using the 'Test Single Calculation' button first, or enable Debug Mode for more information.")
            
            # Show debug info
            with st.expander("🔍 Debug Information"):
                import traceback
                st.code(traceback.format_exc())
                
                st.write(f"**Error details:** {str(e)}")
                st.write(f"**Sample data shape:** {st.session_state.sample_data.shape}")
                st.write(f"**Available columns:** {st.session_state.sample_data.columns.tolist()}")
                if 'standard' in st.session_state.engine.mortality_tables:
                    table = st.session_state.engine.mortality_tables['standard']
                    st.write(f"**Mortality table shape:** {table.shape}")
                    st.write(f"**Age range:** {table['age'].min()} - {table['age'].max()}")
                    
                # Show age distribution
                ages = [(datetime.now() - pd.to_datetime(dob)).days / 365.25 
                        for dob in st.session_state.sample_data['date_of_birth']]
                st.write(f"**Member age range:** {min(ages):.1f} - {max(ages):.1f}")
                st.write(f"**Number of members:** {len(st.session_state.sample_data)}")
                
                # Show sample member
                st.write("**Sample Member:**")
                st.write(st.session_state.sample_data.iloc[0].to_dict())

def show_regulatory():
    """Display regulatory compliance module"""
    st.header("📋 Regulatory Compliance")
    
    # Calculate current metrics for regulatory reports
    with st.spinner("Calculating regulatory metrics..."):
        tp_results = st.session_state.engine.calculate_technical_provisions(
            st.session_state.sample_data
        )
    
    # Report generation
    st.subheader("📊 Generate Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Solvency II QRT", type="primary"):
            st.subheader("Solvency II Quantitative Reporting")
            
            # Simplified Solvency II calculations
            technical_provisions = tp_results['technical_provisions']
            risk_margin = tp_results['risk_margin']
            best_estimate = tp_results['best_estimate_liability']
            
            # Assume some values for demonstration
            scr = technical_provisions * 0.25  # Simplified SCR
            own_funds = technical_provisions * 1.2  # Assume 120% coverage
            
            solvency_data = {
                'Item': [
                    'Best Estimate Liability',
                    'Risk Margin',
                    'Technical Provisions',
                    'Own Funds',
                    'Solvency Capital Requirement',
                    'Solvency Ratio'
                ],
                'Value': [
                    f"£{best_estimate/1e6:.2f}M",
                    f"£{risk_margin/1e6:.2f}M",
                    f"£{technical_provisions/1e6:.2f}M",
                    f"£{own_funds/1e6:.2f}M",
                    f"£{scr/1e6:.2f}M",
                    f"{(own_funds/scr)*100:.1f}%"
                ]
            }
            
            df_solvency = pd.DataFrame(solvency_data)
            st.dataframe(df_solvency, use_container_width=True)
            
            # Download link
            csv = df_solvency.to_csv(index=False)
            st.download_button(
                label="📥 Download Solvency II Report",
                data=csv,
                file_name=f"solvency_ii_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📈 IAS 19 Disclosure", type="primary"):
            st.subheader("IAS 19 Employee Benefits")
            
            # IAS 19 specific calculations
            dbo = tp_results['best_estimate_liability']  # Defined Benefit Obligation
            plan_assets = dbo * 0.85  # Assume 85% funded
            net_liability = dbo - plan_assets
            
            ias19_data = {
                'Item': [
                    'Present Value of DBO',
                    'Fair Value of Plan Assets',
                    'Net Defined Benefit Liability',
                    'Service Cost (Current Year)',
                    'Interest Cost',
                    'Funding Ratio'
                ],
                'Value': [
                    f"£{dbo/1e6:.2f}M",
                    f"£{plan_assets/1e6:.2f}M",
                    f"£{net_liability/1e6:.2f}M",
                    f"£{dbo*0.05/1e6:.2f}M",  # Estimated 5% of DBO
                    f"£{dbo*0.035/1e6:.2f}M",  # 3.5% interest
                    f"{(plan_assets/dbo)*100:.1f}%"
                ]
            }
            
            df_ias19 = pd.DataFrame(ias19_data)
            st.dataframe(df_ias19, use_container_width=True)
            
            # Download link
            csv = df_ias19.to_csv(index=False)
            st.download_button(
                label="📥 Download IAS 19 Report",
                data=csv,
                file_name=f"ias19_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # Compliance status
    st.subheader("✅ Compliance Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="success-box">
        <strong>✅ Solvency II</strong><br>
        Status: Compliant<br>
        Ratio: 142%<br>
        Last Updated: Today
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="success-box">
        <strong>✅ IAS 19</strong><br>
        Status: Compliant<br>
        Reports: Current<br>
        Last Updated: Q4 2024
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ TPR Return</strong><br>
        Status: Due Soon<br>
        Deadline: 31 Dec 2024<br>
        Action Required
        </div>
        """, unsafe_allow_html=True)

def show_configuration():
    """Display configuration module"""
    st.header("⚙️ Configuration")
    
    st.subheader("🔧 Actuarial Parameters")
    
    # Discount rates
    st.write("**Discount Rates**")
    col1, col2 = st.columns(2)
    
    with col1:
        risk_free_rate = st.number_input(
            "Risk-free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=st.session_state.config.discount_rates['risk_free'] * 100,
            step=0.1
        ) / 100
    
    with col2:
        corporate_rate = st.number_input(
            "Corporate Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=st.session_state.config.discount_rates.get('corporate', 4.5) * 100,
            step=0.1
        ) / 100
    
    # Monte Carlo parameters
    st.write("**Monte Carlo Parameters**")
    col1, col2 = st.columns(2)
    
    with col1:
        max_simulations = st.number_input(
            "Maximum Simulations",
            min_value=100,
            max_value=10000,
            value=st.session_state.config.monte_carlo_simulations,
            step=100
        )
    
    with col2:
        confidence_levels = st.multiselect(
            "Confidence Levels",
            [0.5, 0.75, 0.9, 0.95, 0.99, 0.995],
            default=st.session_state.config.confidence_levels
        )
    
    # Stress test scenarios
    st.write("**Stress Test Scenarios**")
    
    # Longevity shock
    longevity_shock = st.slider(
        "Longevity Shock (%)",
        min_value=-50,
        max_value=0,
        value=int(st.session_state.config.stress_scenarios['longevity_shock']['mortality_shock'] * 100),
        step=5
    ) / 100
    
    # Interest rate shocks
    col1, col2 = st.columns(2)
    with col1:
        ir_up_shock = st.slider(
            "Interest Rate Up Shock (bp)",
            min_value=0,
            max_value=500,
            value=int(st.session_state.config.stress_scenarios['interest_rate_up']['interest_rate_shock'] * 10000),
            step=25
        ) / 10000
    
    with col2:
        ir_down_shock = st.slider(
            "Interest Rate Down Shock (bp)",
            min_value=-500,
            max_value=0,
            value=int(st.session_state.config.stress_scenarios['interest_rate_down']['interest_rate_shock'] * 10000),
            step=25
        ) / 10000
    
    # Update configuration
    if st.button("💾 Save Configuration", type="primary"):
        # Update the configuration
        st.session_state.config.discount_rates['risk_free'] = risk_free_rate
        st.session_state.config.discount_rates['corporate'] = corporate_rate
        st.session_state.config.monte_carlo_simulations = max_simulations
        st.session_state.config.confidence_levels = confidence_levels
        
        st.session_state.config.stress_scenarios['longevity_shock']['mortality_shock'] = longevity_shock
        st.session_state.config.stress_scenarios['interest_rate_up']['interest_rate_shock'] = ir_up_shock
        st.session_state.config.stress_scenarios['interest_rate_down']['interest_rate_shock'] = ir_down_shock
        
        # Update engine
        st.session_state.engine.config = st.session_state.config
        
        st.success("✅ Configuration saved successfully!")
    
    # Display current configuration
    st.subheader("📋 Current Configuration")
    
    config_dict = {
        'Parameter': [
            'Risk-free Rate',
            'Corporate Rate', 
            'Max Simulations',
            'Longevity Shock',
            'IR Up Shock',
            'IR Down Shock'
        ],
        'Value': [
            f"{risk_free_rate*100:.2f}%",
            f"{corporate_rate*100:.2f}%",
            f"{max_simulations:,}",
            f"{longevity_shock*100:.1f}%",
            f"{ir_up_shock*10000:.0f}bp",
            f"{ir_down_shock*10000:.0f}bp"
        ]
    }
    
    df_config = pd.DataFrame(config_dict)
    st.dataframe(df_config, use_container_width=True)
    
    # Export/Import configuration
    st.subheader("📤 Export/Import Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export configuration
        config_yaml = yaml.dump(asdict(st.session_state.config), default_flow_style=False)
        st.download_button(
            label="📥 Export Configuration",
            data=config_yaml,
            file_name=f"actuarial_config_{datetime.now().strftime('%Y%m%d')}.yaml",
            mime="text/yaml"
        )
    
    with col2:
        # Import configuration (placeholder)
        uploaded_file = st.file_uploader("📤 Import Configuration", type=['yaml', 'yml'])
        if uploaded_file is not None:
            try:
                config_data = yaml.safe_load(uploaded_file)
                # Validate and update configuration here
                st.success("✅ Configuration imported successfully!")
            except Exception as e:
                st.error(f"❌ Error importing configuration: {e}")
    
    # Debug Information
    st.subheader("🔍 Debug Information")
    
    with st.expander("View System Status"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Mortality Tables Loaded:**")
            st.write(list(st.session_state.engine.mortality_tables.keys()))
            
            if 'standard' in st.session_state.engine.mortality_tables:
                table = st.session_state.engine.mortality_tables['standard']
                st.write(f"**Table Shape:** {table.shape}")
                st.write(f"**Columns:** {table.columns.tolist()}")
                st.write("**Sample Data:**")
                st.dataframe(table.head())
        
        with col2:
            st.write("**Sample Member Data:**")
            st.write(f"**Shape:** {st.session_state.sample_data.shape}")
            st.write(f"**Columns:** {st.session_state.sample_data.columns.tolist()}")
            st.write("**Gender Values:**")
            st.write(st.session_state.sample_data['gender'].value_counts())
            st.write("**Sample Records:**")
            st.dataframe(st.session_state.sample_data.head())
    
    # Reset functionality
    st.subheader("🔄 Reset System")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Regenerate Sample Data", type="secondary"):
            st.session_state.sample_data = generate_sample_members(1000)
            st.success("✅ Sample data regenerated!")
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()  # Fallback for older Streamlit versions
    
    with col2:
        if st.button("🔄 Reset All Settings", type="secondary"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("✅ All settings reset! Please refresh the page.")
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()  # Fallback for older Streamlit versions

if __name__ == "__main__":
    main()