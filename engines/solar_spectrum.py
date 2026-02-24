#!/usr/bin/env python3
"""
Realistic Solar Spectrum Model
==============================

Module for calculating realistic solar spectra considering:
- Air mass variation with solar angle
- Latitude and time dependencies
- Atmospheric absorption and scattering
- Temperature and humidity effects
- Sunrise/sunset calculations

Features:
- Air mass calculation: AM = 1/cos(θ_zenith)
- Spectral modification with atmospheric path length
- Daily/seasonal solar geometry
- Simple parametric atmospheric model (no full SMARTS needed)

References:
- Bird & Hulstrom (1981) - Simple solar spectral model
- Iqbal (1983) - Introduction to Solar Radiation
- NREL Solar Position Algorithm (SPA)
- Kasten & Young (1989) - Air mass formula
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

@dataclass
class SolarConditions:
    """Solar irradiance conditions at specific time/location"""
    timestamp: datetime
    latitude: float  # degrees
    longitude: float  # degrees
    air_mass: float
    zenith_angle: float  # degrees
    azimuth_angle: float  # degrees
    is_daylight: bool
    dni: float  # Direct Normal Irradiance [W/m²]
    dhi: float  # Diffuse Horizontal Irradiance [W/m²]
    ghi: float  # Global Horizontal Irradiance [W/m²]

@dataclass
class SpectralData:
    """Spectral irradiance data"""
    wavelengths: np.ndarray  # nm
    irradiance: np.ndarray   # W⋅m⁻²⋅nm⁻¹
    air_mass: float
    total_irradiance: float  # W/m²

class RealisticSolarSpectrum:
    """
    Realistic solar spectrum calculator with atmospheric effects.
    """
    
    def __init__(self):
        # Standard wavelength range
        self.wavelength_range = np.linspace(280, 2500, 445)  # 5nm resolution
        
        # Solar constant and extraterrestrial spectrum
        self.solar_constant = 1361.0  # W/m² (updated value)
        
        # Atmospheric absorption coefficients
        # Simplified model based on major absorbers
        self.absorption_coeffs = self._init_absorption_model()
        
        # Rayleigh scattering coefficient (wavelength⁻⁴ dependence)
        self.rayleigh_coeff = 8.7e-5  # at 1000 nm reference
        
        # Standard temperature and pressure
        self.std_pressure = 1013.25  # mbar
        self.std_temperature = 288.15  # K (15°C)
    
    def _init_absorption_model(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Initialize simplified atmospheric absorption model.
        
        Returns dictionary with (wavelength, absorption_coeff) for major absorbers.
        """
        
        wavelengths = self.wavelength_range
        
        # Water vapor absorption bands (approximate)
        h2o_absorption = np.zeros_like(wavelengths)
        
        # Major water bands: 940nm, 1130nm, 1400nm, 1900nm
        h2o_bands = [
            (940, 50, 0.3),   # (center_nm, width_nm, strength)
            (1130, 40, 0.2),
            (1400, 80, 0.8),
            (1900, 200, 1.5),
            (2700, 300, 2.0)  # Strong NIR absorption
        ]
        
        for center, width, strength in h2o_bands:
            h2o_absorption += strength * np.exp(-0.5 * ((wavelengths - center) / width)**2)
        
        # Oxygen absorption (A and B bands)
        o2_absorption = np.zeros_like(wavelengths)
        o2_bands = [(760, 15, 0.8), (1270, 30, 0.4)]  # A and B bands
        
        for center, width, strength in o2_bands:
            o2_absorption += strength * np.exp(-0.5 * ((wavelengths - center) / width)**2)
        
        # Ozone absorption (Hartley and Huggins bands)
        o3_absorption = np.zeros_like(wavelengths)
        # Strong UV absorption below 350nm
        uv_mask = wavelengths < 350
        o3_absorption[uv_mask] = 2.0 * np.exp(-(wavelengths[uv_mask] - 280) / 30)
        
        # Weak visible absorption around 600nm (Chappuis band)
        o3_absorption += 0.1 * np.exp(-0.5 * ((wavelengths - 600) / 100)**2)
        
        # CO₂ absorption bands (weak but present)
        co2_absorption = np.zeros_like(wavelengths)
        co2_bands = [(2000, 50, 0.1), (2700, 100, 0.2)]
        
        for center, width, strength in co2_bands:
            co2_absorption += strength * np.exp(-0.5 * ((wavelengths - center) / width)**2)
        
        return {
            'H2O': (wavelengths, h2o_absorption),
            'O2': (wavelengths, o2_absorption),
            'O3': (wavelengths, o3_absorption),
            'CO2': (wavelengths, co2_absorption)
        }
    
    def calculate_air_mass(self, zenith_angle: float, altitude: float = 0) -> float:
        """
        Calculate air mass using Kasten-Young formula.
        
        AM = 1 / [cos(θ) + 0.50572(96.07995 - θ)⁻¹·⁶³⁶⁴]
        
        Args:
            zenith_angle: Solar zenith angle [degrees]
            altitude: Observer altitude [m above sea level]
            
        Returns:
            Air mass (dimensionless)
        """
        
        if zenith_angle >= 90:
            return float('inf')  # Sun below horizon
        
        zenith_rad = np.radians(zenith_angle)
        
        # Kasten-Young formula (more accurate than simple 1/cos(θ))
        if zenith_angle < 80:
            am = 1.0 / (np.cos(zenith_rad) + 
                       0.50572 * (96.07995 - zenith_angle)**(-1.6364))
        else:
            # Simple formula for large angles
            am = 1.0 / np.cos(zenith_rad)
        
        # Altitude correction (approximate)
        pressure_ratio = np.exp(-altitude / 8400)  # Scale height ≈ 8.4 km
        am *= pressure_ratio
        
        return max(am, 1.0)  # Minimum AM = 1.0
    
    def solar_position(self, latitude: float, day_of_year: int, hour: float) -> Tuple[float, float]:
        """
        Calculate solar zenith and azimuth angles.
        
        Uses simplified solar position algorithm suitable for PV calculations.
        
        Args:
            latitude: Observer latitude [degrees]
            day_of_year: Day of year (1-365)
            hour: Hour of day (0-24, decimal allowed)
            
        Returns:
            (zenith_angle, azimuth_angle) in degrees
        """
        
        lat_rad = np.radians(latitude)
        
        # Solar declination (Cooper's equation)
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        decl_rad = np.radians(declination)
        
        # Hour angle
        hour_angle = 15 * (hour - 12)  # degrees
        hour_rad = np.radians(hour_angle)
        
        # Solar elevation angle
        sin_elevation = (np.sin(lat_rad) * np.sin(decl_rad) + 
                        np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_rad))
        
        elevation = np.degrees(np.arcsin(np.clip(sin_elevation, -1, 1)))
        zenith_angle = 90 - elevation
        
        # Solar azimuth angle (from south, clockwise)
        cos_azimuth = ((np.sin(decl_rad) * np.cos(lat_rad) - 
                       np.cos(decl_rad) * np.sin(lat_rad) * np.cos(hour_rad)) / 
                      np.cos(np.radians(elevation + 1e-10)))  # Avoid division by zero
        
        azimuth = np.degrees(np.arccos(np.clip(cos_azimuth, -1, 1)))
        
        # Correct quadrant for azimuth
        if hour_angle > 0:  # Afternoon
            azimuth = 360 - azimuth
        
        return max(zenith_angle, 0), azimuth % 360
    
    def sunrise_sunset_times(self, latitude: float, day_of_year: int) -> Tuple[float, float]:
        """
        Calculate sunrise and sunset times.
        
        Args:
            latitude: Observer latitude [degrees]
            day_of_year: Day of year (1-365)
            
        Returns:
            (sunrise_hour, sunset_hour) in decimal hours
        """
        
        lat_rad = np.radians(latitude)
        
        # Solar declination
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        decl_rad = np.radians(declination)
        
        # Hour angle at sunrise/sunset (when elevation = 0°)
        try:
            cos_hour_angle = -np.tan(lat_rad) * np.tan(decl_rad)
            
            if cos_hour_angle > 1:
                # Polar night
                return 12.0, 12.0  # No daylight
            elif cos_hour_angle < -1:
                # Midnight sun
                return 0.0, 24.0  # 24h daylight
            
            hour_angle = np.degrees(np.arccos(cos_hour_angle))
            
            sunrise = 12 - hour_angle / 15
            sunset = 12 + hour_angle / 15
            
            return sunrise, sunset
            
        except:
            return 6.0, 18.0  # Default 12h day
    
    def extraterrestrial_spectrum(self, air_mass: float = 0) -> SpectralData:
        """
        Calculate extraterrestrial solar spectrum (AM0).
        
        Uses simplified blackbody model adjusted to match solar spectrum.
        """
        
        wavelengths = self.wavelength_range
        wavelength_m = wavelengths * 1e-9
        
        # Solar constants
        T_sun = 5778  # K
        h = 6.62607015e-34  # J⋅s
        c = 2.99792458e8    # m/s
        k = 1.380649e-23    # J/K
        
        # Blackbody spectrum (Planck function)
        planck = (2 * h * c**2 / wavelength_m**5) / (np.exp(h * c / (wavelength_m * k * T_sun)) - 1)
        
        # Convert from W⋅m⁻²⋅m⁻¹ to W⋅m⁻²⋅nm⁻¹
        irradiance_nm = planck * 1e-9
        
        # Normalize to solar constant
        total_irradiance = np.trapezoid(irradiance_nm, wavelengths)
        if total_irradiance > 0:
            irradiance_nm *= self.solar_constant / total_irradiance
        
        return SpectralData(
            wavelengths=wavelengths,
            irradiance=irradiance_nm,
            air_mass=0.0,
            total_irradiance=self.solar_constant
        )
    
    def atmospheric_transmission(self, air_mass: float, 
                               relative_humidity: float = 50.0,
                               altitude: float = 0.0) -> np.ndarray:
        """
        Calculate atmospheric transmission spectrum.
        
        Args:
            air_mass: Air mass
            relative_humidity: Relative humidity [%]
            altitude: Altitude [m above sea level]
            
        Returns:
            Transmission spectrum (0-1)
        """
        
        wavelengths = self.wavelength_range
        transmission = np.ones_like(wavelengths)
        
        # Pressure correction for altitude
        pressure_ratio = np.exp(-altitude / 8400)
        effective_am = air_mass * pressure_ratio
        
        # Rayleigh scattering (wavelength⁻⁴ dependence)
        rayleigh_optical_depth = self.rayleigh_coeff * (1000 / wavelengths)**4 * effective_am
        rayleigh_transmission = np.exp(-rayleigh_optical_depth)
        
        transmission *= rayleigh_transmission
        
        # Molecular absorption
        for species, (wl, absorption) in self.absorption_coeffs.items():
            
            # Scale absorption by air mass and conditions
            if species == 'H2O':
                # Water vapor scales with humidity
                scale_factor = (relative_humidity / 50.0) * effective_am
            elif species == 'O3':
                # Ozone is mostly in stratosphere, less AM dependence
                scale_factor = effective_am**0.5
            else:
                # Other species scale linearly with air mass
                scale_factor = effective_am
            
            # Interpolate to common wavelength grid if needed
            if len(wl) != len(wavelengths):
                absorption_interp = np.interp(wavelengths, wl, absorption)
            else:
                absorption_interp = absorption
            
            optical_depth = absorption_interp * scale_factor
            molecular_transmission = np.exp(-optical_depth)
            
            transmission *= molecular_transmission
        
        # Aerosol scattering (simplified)
        # Assume Ångström turbidity with typical values
        beta_500 = 0.1  # Turbidity coefficient at 500nm
        alpha = 1.3     # Ångström exponent
        
        aerosol_optical_depth = beta_500 * (wavelengths / 500)**(-alpha) * effective_am
        aerosol_transmission = np.exp(-aerosol_optical_depth)
        
        transmission *= aerosol_transmission
        
        # Ensure transmission stays in [0,1]
        transmission = np.clip(transmission, 0, 1)
        
        return transmission
    
    def calculate_spectrum(self, air_mass: float, 
                          relative_humidity: float = 50.0,
                          altitude: float = 0.0) -> SpectralData:
        """
        Calculate realistic solar spectrum for given conditions.
        
        Args:
            air_mass: Air mass
            relative_humidity: Relative humidity [%]
            altitude: Observer altitude [m]
            
        Returns:
            SpectralData with realistic spectrum
        """
        
        # Get extraterrestrial spectrum
        am0_spectrum = self.extraterrestrial_spectrum()
        
        # Calculate atmospheric transmission
        transmission = self.atmospheric_transmission(air_mass, relative_humidity, altitude)
        
        # Apply transmission to extraterrestrial spectrum
        surface_irradiance = am0_spectrum.irradiance * transmission
        
        # Calculate total irradiance
        total_irradiance = np.trapezoid(surface_irradiance, am0_spectrum.wavelengths)
        
        return SpectralData(
            wavelengths=am0_spectrum.wavelengths,
            irradiance=surface_irradiance,
            air_mass=air_mass,
            total_irradiance=total_irradiance
        )
    
    def daily_irradiance_profile(self, latitude: float, day_of_year: int,
                                relative_humidity: float = 50.0,
                                altitude: float = 0.0,
                                time_resolution: float = 0.5) -> pd.DataFrame:
        """
        Calculate daily irradiance profile for given location and date.
        
        Args:
            latitude: Observer latitude [degrees]
            day_of_year: Day of year (1-365)
            relative_humidity: Relative humidity [%]
            altitude: Observer altitude [m]
            time_resolution: Time step [hours]
            
        Returns:
            DataFrame with hourly solar data
        """
        
        # Time array
        hours = np.arange(0, 24 + time_resolution, time_resolution)
        
        data = []
        
        for hour in hours:
            # Solar position
            zenith, azimuth = self.solar_position(latitude, day_of_year, hour)
            
            # Check if sun is above horizon
            is_daylight = zenith < 90
            
            if is_daylight:
                # Calculate air mass and spectrum
                air_mass = self.calculate_air_mass(zenith, altitude)
                spectrum = self.calculate_spectrum(air_mass, relative_humidity, altitude)
                
                # Estimate direct/diffuse split (simplified)
                clearness_index = min(spectrum.total_irradiance / 1000, 1.0)
                
                if clearness_index > 0.7:
                    # Clear sky - high direct component
                    dni = spectrum.total_irradiance * 0.9 / max(np.cos(np.radians(zenith)), 0.1)
                    dhi = spectrum.total_irradiance * 0.1
                elif clearness_index > 0.3:
                    # Partly cloudy
                    dni = spectrum.total_irradiance * 0.6 / max(np.cos(np.radians(zenith)), 0.1)
                    dhi = spectrum.total_irradiance * 0.4
                else:
                    # Cloudy - mostly diffuse
                    dni = spectrum.total_irradiance * 0.2 / max(np.cos(np.radians(zenith)), 0.1)
                    dhi = spectrum.total_irradiance * 0.8
                
                ghi = spectrum.total_irradiance
                
            else:
                air_mass = float('inf')
                dni = dhi = ghi = 0
                spectrum = None
            
            data.append({
                'hour': hour,
                'zenith_angle': zenith,
                'azimuth_angle': azimuth,
                'air_mass': air_mass if air_mass != float('inf') else 0,
                'is_daylight': is_daylight,
                'dni': dni,
                'dhi': dhi,
                'ghi': ghi,
                'total_irradiance': ghi
            })
        
        return pd.DataFrame(data)
    
    def seasonal_comparison(self, latitude: float, 
                           days: List[int] = [172, 266, 355],  # Summer/Fall/Winter solstices + equinox
                           relative_humidity: float = 50.0) -> Dict[str, pd.DataFrame]:
        """
        Compare solar irradiance across seasons.
        
        Args:
            latitude: Observer latitude [degrees]
            days: Days of year to compare
            relative_humidity: Relative humidity [%]
            
        Returns:
            Dictionary mapping season names to daily profiles
        """
        
        season_names = ['Summer Solstice', 'Equinox', 'Winter Solstice']
        if len(days) > 3:
            season_names = [f'Day {d}' for d in days]
        
        results = {}
        
        for i, day in enumerate(days):
            season_name = season_names[i] if i < len(season_names) else f'Day {day}'
            profile = self.daily_irradiance_profile(latitude, day, relative_humidity)
            results[season_name] = profile
        
        return results
    
    def temperature_profile(self, latitude: float, day_of_year: int,
                           daily_temp_range: float = 20.0,
                           min_temp: float = 15.0) -> pd.DataFrame:
        """
        Generate simple sinusoidal temperature profile.
        
        Args:
            latitude: Observer latitude [degrees]  
            day_of_year: Day of year (1-365)
            daily_temp_range: Temperature swing [°C]
            min_temp: Minimum daily temperature [°C]
            
        Returns:
            DataFrame with hourly temperature data
        """
        
        hours = np.arange(0, 25, 0.5)  # Include 24.0 for full cycle
        
        # Simple sinusoidal model: minimum at 6 AM, maximum at 2 PM
        temp_celsius = (min_temp + daily_temp_range/2 + 
                       (daily_temp_range/2) * np.sin(2*np.pi*(hours - 6)/24))
        
        # Seasonal adjustment
        seasonal_offset = 10 * np.cos(2*np.pi*(day_of_year - 172)/365)  # Peak at summer solstice
        temp_celsius += seasonal_offset
        
        # Latitude effect (rough approximation)
        lat_factor = 1 - abs(latitude)/90
        temp_celsius *= lat_factor
        
        df = pd.DataFrame({
            'hour': hours[:-1],  # Remove duplicate 24.0
            'temperature_c': temp_celsius[:-1],
            'temperature_k': temp_celsius[:-1] + 273.15
        })
        
        return df

# Global instance
SOLAR_SPECTRUM_CALCULATOR = RealisticSolarSpectrum()

if __name__ == "__main__":
    print("Realistic Solar Spectrum Test")
    print("=" * 40)
    
    calc = RealisticSolarSpectrum()
    
    # Test spectrum calculation
    spectrum_am1 = calc.calculate_spectrum(air_mass=1.0)
    spectrum_am2 = calc.calculate_spectrum(air_mass=2.0)
    
    print(f"AM1.0 total irradiance: {spectrum_am1.total_irradiance:.1f} W/m²")
    print(f"AM2.0 total irradiance: {spectrum_am2.total_irradiance:.1f} W/m²")
    
    # Test daily profile
    latitude = 37.5  # Seoul
    day_of_year = 172  # Summer solstice
    
    daily_profile = calc.daily_irradiance_profile(latitude, day_of_year)
    max_irradiance = daily_profile['ghi'].max()
    daily_energy = daily_profile['ghi'].sum() * 0.5  # Wh/m² (0.5h intervals)
    
    print(f"\nDaily profile (Seoul, summer solstice):")
    print(f"Peak irradiance: {max_irradiance:.1f} W/m²")
    print(f"Daily energy: {daily_energy/1000:.2f} kWh/m²")
    
    # Test sunrise/sunset
    sunrise, sunset = calc.sunrise_sunset_times(latitude, day_of_year)
    print(f"Sunrise: {sunrise:.2f}h, Sunset: {sunset:.2f}h")
    print(f"Daylight hours: {sunset - sunrise:.1f}h")