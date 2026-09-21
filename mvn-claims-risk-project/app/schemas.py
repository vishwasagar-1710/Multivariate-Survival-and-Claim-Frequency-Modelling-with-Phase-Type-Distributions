"""Pydantic schemas for the risk-scoring API (FastAPI)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PolicyInput(BaseModel):
    Area: str = Field(..., description="Rating area code A-F", examples=["C"])
    VehPower: int = Field(..., ge=1, le=20)
    VehAge: int = Field(..., ge=0, le=60)
    DrivAge: int = Field(..., ge=18, le=100)
    BonusMalus: float = Field(..., ge=50, le=350)
    Density: float = Field(..., ge=1, description="Population density of the policyholder's commune")
    Region: str = Field(..., examples=["R3"])
    VehGas: str = Field(..., examples=["Diesel", "Regular"])
    VehBrand: str = Field(..., examples=["B4"])
    Exposure: float = Field(..., gt=0, le=1.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "Area": "C", "VehPower": 7, "VehAge": 4, "DrivAge": 28,
                "BonusMalus": 95, "Density": 3200, "Region": "R3",
                "VehGas": "Diesel", "VehBrand": "B4", "Exposure": 0.85,
            }
        }
    }


class BatchPolicyInput(BaseModel):
    policies: list[PolicyInput]


class RiskScoreOutput(BaseModel):
    high_risk_probability: float
    high_risk_flag: bool
    risk_tier: str
    operating_threshold: float
    model_version: str


class BatchRiskScoreOutput(BaseModel):
    results: list[RiskScoreOutput]
