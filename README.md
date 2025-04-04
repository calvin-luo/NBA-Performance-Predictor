# NBA-Performance-Predictor (WIP)
by: Calvin Luo

## Objective
A web application that predicts NBA game outcomes based on players' past performances.

## Data Flow
```mermaid
graph TD
    subgraph "Data Collection Layer"
        A1[NBA API] -->|Game Schedules| B1[Game Scraper]
        A2[Rotowire.com] -->|Lineups & Player Status| B2[Player Scraper]
    end

    subgraph "Data Storage Layer"
        B1 -->|Store Games| C[SQLite Database]
        B2 -->|Store Lineups| C
    end

    subgraph "Data Analysis Layer"
        C -->|Retrieve Lineups| D[Player Stats Collector]
        D -->|Retrieve Player History| E1[NBA API]
        E1 -->|Return Historical Stats| D
        D -->|Provide Processed Stats| F[Time Series Models]
    end

    subgraph "Prediction Layer"
        F -->|ARIMA/SARIMA Analysis| G[Game Predictions]
    end