import requests
import json
from typing import Dict, Any, Optional
from google.adk.tools import ToolContext

# Financial Modeling Prep API Configuration
FMP_API_KEY = "h5bDd4wwF9CeBjtcKdTkh7IwnOEC8jPT"
FMP_BASE_URL = "https://financialmodelingprep.com/stable"

def company_profile_tool(symbol: str, tool_context: ToolContext) -> str:
    """
    Retrieves comprehensive company profile information for a given stock symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
        tool_context: The context for the tool call
        
    Returns:
        JSON string containing company profile data including company name, industry, 
        sector, market cap, description, and other key metrics.
    """
    print(f"--- TOOL: company_profile_tool called for symbol: {symbol} ---")
    
    try:
        url = f"{FMP_BASE_URL}/profile"
        params = {
            "symbol": symbol.upper(),
            "apikey": FMP_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return f"No company profile data found for symbol: {symbol}"
        
        # Extract key information for better readability
        if isinstance(data, list) and len(data) > 0:
            profile = data[0]
            
            summary = {
                "symbol": profile.get("symbol", "N/A"),
                "company_name": profile.get("companyName", "N/A"),
                "price": profile.get("price", "N/A"),
                "market_cap": profile.get("mktCap", "N/A"),
                "industry": profile.get("industry", "N/A"),
                "sector": profile.get("sector", "N/A"),
                "description": profile.get("description", "N/A"),
                "ceo": profile.get("ceo", "N/A"),
                "exchange": profile.get("exchangeShortName", "N/A"),
                "website": profile.get("website", "N/A"),
                "full_profile": profile
            }
            
            return json.dumps(summary, indent=2)
        else:
            return json.dumps(data, indent=2)
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Error fetching company profile for {symbol}: {e}"
        print(error_msg)
        return error_msg
    except json.JSONDecodeError as e:
        error_msg = f"Error parsing company profile response for {symbol}: {e}"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Unexpected error in company_profile_tool for {symbol}: {e}"
        print(error_msg)
        return error_msg


def price_target_summary_tool(symbol: str, tool_context: ToolContext) -> str:
    """
    Retrieves analyst price target summary for a given stock symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
        tool_context: The context for the tool call
        
    Returns:
        JSON string containing price target data including average target price,
        number of analysts, high/low targets, and other analyst sentiment data.
    """
    print(f"--- TOOL: price_target_summary_tool called for symbol: {symbol} ---")
    
    try:
        url = f"{FMP_BASE_URL}/price-target-summary"
        params = {
            "symbol": symbol.upper(),
            "apikey": FMP_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return f"No price target data found for symbol: {symbol}"
        
        # Extract key information for better readability
        if isinstance(data, list) and len(data) > 0:
            target_data = data[0]
            
            summary = {
                "symbol": target_data.get("symbol", "N/A"),
                "last_month_avg_price_target": target_data.get("lastMonthAvgPriceTarget", "N/A"),
                "last_quarter_avg_price_target": target_data.get("lastQuarterAvgPriceTarget", "N/A"),
                "last_year_avg_price_target": target_data.get("lastYearAvgPriceTarget", "N/A"),
                "all_time_avg_price_target": target_data.get("allTimeAvgPriceTarget", "N/A"),
                "last_month_high_target": target_data.get("lastMonthHighPriceTarget", "N/A"),
                "last_month_low_target": target_data.get("lastMonthLowPriceTarget", "N/A"),
                "number_of_analysts": target_data.get("numberOfAnalysts", "N/A"),
                "full_data": target_data
            }
            
            return json.dumps(summary, indent=2)
        else:
            return json.dumps(data, indent=2)
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Error fetching price targets for {symbol}: {e}"
        print(error_msg)
        return error_msg
    except json.JSONDecodeError as e:
        error_msg = f"Error parsing price target response for {symbol}: {e}"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Unexpected error in price_target_summary_tool for {symbol}: {e}"
        print(error_msg)
        return error_msg


def stock_chart_5min_tool(symbol: str, from_date: Optional[str] = None, 
                         to_date: Optional[str] = None, nonadjusted: bool = False, 
                         tool_context: ToolContext = None) -> str:
    """
    Retrieves 5-minute interval stock chart data for a given symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
        from_date: Start date in YYYY-MM-DD format (optional)
        to_date: End date in YYYY-MM-DD format (optional)
        nonadjusted: Whether to get non-adjusted prices (default: False)
        tool_context: The context for the tool call
        
    Returns:
        JSON string containing 5-minute interval price data including open, high, 
        low, close, and volume for each 5-minute period.
    """
    print(f"--- TOOL: stock_chart_5min_tool called for symbol: {symbol} ---")
    
    try:
        url = f"{FMP_BASE_URL}/historical-chart/5min"
        params = {
            "symbol": symbol.upper(),
            "apikey": FMP_API_KEY
        }
        
        # Add optional parameters if provided
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        if nonadjusted:
            params["nonadjusted"] = str(nonadjusted).lower()
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return f"No 5-minute chart data found for symbol: {symbol}"
        
        # Limit the response to avoid overwhelming output (last 50 data points)
        if isinstance(data, list) and len(data) > 50:
            limited_data = data[-50:]  # Get the most recent 50 data points
            summary = {
                "symbol": symbol.upper(),
                "total_data_points": len(data),
                "showing_recent_points": len(limited_data),
                "date_range": f"{data[0].get('date', 'N/A')} to {data[-1].get('date', 'N/A')}" if data else "N/A",
                "parameters": {
                    "from_date": from_date,
                    "to_date": to_date,
                    "nonadjusted": nonadjusted
                },
                "recent_data": limited_data
            }
        else:
            summary = {
                "symbol": symbol.upper(),
                "total_data_points": len(data) if isinstance(data, list) else 1,
                "parameters": {
                    "from_date": from_date,
                    "to_date": to_date,
                    "nonadjusted": nonadjusted
                },
                "data": data
            }
        
        return json.dumps(summary, indent=2)
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Error fetching 5-min chart data for {symbol}: {e}"
        print(error_msg)
        return error_msg
    except json.JSONDecodeError as e:
        error_msg = f"Error parsing 5-min chart response for {symbol}: {e}"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Unexpected error in stock_chart_5min_tool for {symbol}: {e}"
        print(error_msg)
        return error_msg


def income_statement_tool(symbol: str, period: str = "annual", limit: int = 5, tool_context: ToolContext = None) -> str:
    """
    Retrieves income statement data for a given stock symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
        period: 'annual' or 'quarter' (default: 'annual')
        limit: Number of periods to retrieve (default: 5)
        tool_context: The context for the tool call
        
    Returns:
        JSON string containing income statement data including revenue, expenses,
        net income, EPS, and other profitability metrics.
    """
    print(f"--- TOOL: income_statement_tool called for symbol: {symbol}, period: {period} ---")
    
    try:
        url = f"{FMP_BASE_URL}/income-statement"
        params = {
            "symbol": symbol.upper(),
            "period": period,
            "limit": limit,
            "apikey": FMP_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return f"No income statement data found for symbol: {symbol}"
        
        # Extract key metrics for better readability
        if isinstance(data, list) and len(data) > 0:
            summary = {
                "symbol": symbol.upper(),
                "period_type": period,
                "periods_retrieved": len(data),
                "latest_period": data[0].get("date", "N/A") if data else "N/A",
                "key_metrics_latest": {
                    "revenue": data[0].get("revenue", "N/A"),
                    "gross_profit": data[0].get("grossProfit", "N/A"),
                    "operating_income": data[0].get("operatingIncome", "N/A"),
                    "net_income": data[0].get("netIncome", "N/A"),
                    "eps": data[0].get("eps", "N/A"),
                    "eps_diluted": data[0].get("epsdiluted", "N/A")
                } if data else {},
                "full_statements": data
            }
            
            return json.dumps(summary, indent=2)
        else:
            return json.dumps(data, indent=2)
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Error fetching income statement for {symbol}: {e}"
        print(error_msg)
        return error_msg
    except json.JSONDecodeError as e:
        error_msg = f"Error parsing income statement response for {symbol}: {e}"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Unexpected error in income_statement_tool for {symbol}: {e}"
        print(error_msg)
        return error_msg


def balance_sheet_tool(symbol: str, period: str = "annual", limit: int = 5, tool_context: ToolContext = None) -> str:
    """
    Retrieves balance sheet data for a given stock symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
        period: 'annual' or 'quarter' (default: 'annual')
        limit: Number of periods to retrieve (default: 5)
        tool_context: The context for the tool call
        
    Returns:
        JSON string containing balance sheet data including assets, liabilities,
        and shareholders' equity.
    """
    print(f"--- TOOL: balance_sheet_tool called for symbol: {symbol}, period: {period} ---")
    
    try:
        url = f"{FMP_BASE_URL}/balance-sheet-statement"
        params = {
            "symbol": symbol.upper(),
            "period": period,
            "limit": limit,
            "apikey": FMP_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return f"No balance sheet data found for symbol: {symbol}"
        
        # Extract key metrics for better readability
        if isinstance(data, list) and len(data) > 0:
            summary = {
                "symbol": symbol.upper(),
                "period_type": period,
                "periods_retrieved": len(data),
                "latest_period": data[0].get("date", "N/A") if data else "N/A",
                "key_metrics_latest": {
                    "total_assets": data[0].get("totalAssets", "N/A"),
                    "total_liabilities": data[0].get("totalLiabilities", "N/A"),
                    "total_equity": data[0].get("totalEquity", "N/A"),
                    "cash_and_equivalents": data[0].get("cashAndCashEquivalents", "N/A"),
                    "total_debt": data[0].get("totalDebt", "N/A"),
                    "working_capital": data[0].get("netReceivables", "N/A")
                } if data else {},
                "full_statements": data
            }
            
            return json.dumps(summary, indent=2)
        else:
            return json.dumps(data, indent=2)
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Error fetching balance sheet for {symbol}: {e}"
        print(error_msg)
        return error_msg
    except json.JSONDecodeError as e:
        error_msg = f"Error parsing balance sheet response for {symbol}: {e}"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Unexpected error in balance_sheet_tool for {symbol}: {e}"
        print(error_msg)
        return error_msg


def cash_flow_tool(symbol: str, period: str = "annual", limit: int = 5, tool_context: ToolContext = None) -> str:
    """
    Retrieves cash flow statement data for a given stock symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
        period: 'annual' or 'quarter' (default: 'annual')
        limit: Number of periods to retrieve (default: 5)
        tool_context: The context for the tool call
        
    Returns:
        JSON string containing cash flow data including operating, investing,
        and financing cash flows.
    """
    print(f"--- TOOL: cash_flow_tool called for symbol: {symbol}, period: {period} ---")
    
    try:
        url = f"{FMP_BASE_URL}/cash-flow-statement"
        params = {
            "symbol": symbol.upper(),
            "period": period,
            "limit": limit,
            "apikey": FMP_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return f"No cash flow data found for symbol: {symbol}"
        
        # Extract key metrics for better readability
        if isinstance(data, list) and len(data) > 0:
            summary = {
                "symbol": symbol.upper(),
                "period_type": period,
                "periods_retrieved": len(data),
                "latest_period": data[0].get("date", "N/A") if data else "N/A",
                "key_metrics_latest": {
                    "operating_cash_flow": data[0].get("operatingCashFlow", "N/A"),
                    "investing_cash_flow": data[0].get("netCashUsedForInvestingActivites", "N/A"),
                    "financing_cash_flow": data[0].get("netCashUsedProvidedByFinancingActivities", "N/A"),
                    "free_cash_flow": data[0].get("freeCashFlow", "N/A"),
                    "net_change_in_cash": data[0].get("netChangeInCash", "N/A"),
                    "capex": data[0].get("capitalExpenditure", "N/A")
                } if data else {},
                "full_statements": data
            }
            
            return json.dumps(summary, indent=2)
        else:
            return json.dumps(data, indent=2)
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Error fetching cash flow for {symbol}: {e}"
        print(error_msg)
        return error_msg
    except json.JSONDecodeError as e:
        error_msg = f"Error parsing cash flow response for {symbol}: {e}"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Unexpected error in cash_flow_tool for {symbol}: {e}"
        print(error_msg)
        return error_msg


def key_metrics_tool(symbol: str, period: str = "annual", limit: int = 5, tool_context: ToolContext = None) -> str:
    """
    Retrieves key financial metrics for a given stock symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
        period: 'annual' or 'quarter' (default: 'annual')
        limit: Number of periods to retrieve (default: 5)
        tool_context: The context for the tool call
        
    Returns:
        JSON string containing key financial metrics including P/E ratio,
        market cap, ROE, debt ratios, and other performance indicators.
    """
    print(f"--- TOOL: key_metrics_tool called for symbol: {symbol}, period: {period} ---")
    
    try:
        url = f"{FMP_BASE_URL}/key-metrics"
        params = {
            "symbol": symbol.upper(),
            "period": period,
            "limit": limit,
            "apikey": FMP_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return f"No key metrics data found for symbol: {symbol}"
        
        # Extract key metrics for better readability
        if isinstance(data, list) and len(data) > 0:
            summary = {
                "symbol": symbol.upper(),
                "period_type": period,
                "periods_retrieved": len(data),
                "latest_period": data[0].get("date", "N/A") if data else "N/A",
                "key_metrics_latest": {
                    "market_cap": data[0].get("marketCap", "N/A"),
                    "pe_ratio": data[0].get("peRatio", "N/A"),
                    "price_to_book": data[0].get("priceToBookRatio", "N/A"),
                    "price_to_sales": data[0].get("priceToSalesRatio", "N/A"),
                    "roe": data[0].get("roe", "N/A"),
                    "roa": data[0].get("roa", "N/A"),
                    "debt_to_equity": data[0].get("debtToEquity", "N/A"),
                    "current_ratio": data[0].get("currentRatio", "N/A"),
                    "revenue_per_share": data[0].get("revenuePerShare", "N/A"),
                    "book_value_per_share": data[0].get("bookValuePerShare", "N/A")
                } if data else {},
                "full_metrics": data
            }
            
            return json.dumps(summary, indent=2)
        else:
            return json.dumps(data, indent=2)
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Error fetching key metrics for {symbol}: {e}"
        print(error_msg)
        return error_msg
    except json.JSONDecodeError as e:
        error_msg = f"Error parsing key metrics response for {symbol}: {e}"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Unexpected error in key_metrics_tool for {symbol}: {e}"
        print(error_msg)
        return error_msg


def financial_ratios_tool(symbol: str, period: str = "annual", limit: int = 5, tool_context: ToolContext = None) -> str:
    """
    Retrieves detailed financial ratios for a given stock symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
        period: 'annual' or 'quarter' (default: 'annual')
        limit: Number of periods to retrieve (default: 5)
        tool_context: The context for the tool call
        
    Returns:
        JSON string containing comprehensive financial ratios including liquidity,
        profitability, leverage, and efficiency ratios.
    """
    print(f"--- TOOL: financial_ratios_tool called for symbol: {symbol}, period: {period} ---")
    
    try:
        url = f"{FMP_BASE_URL}/ratios"
        params = {
            "symbol": symbol.upper(),
            "period": period,
            "limit": limit,
            "apikey": FMP_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return f"No financial ratios data found for symbol: {symbol}"
        
        # Extract key ratios for better readability
        if isinstance(data, list) and len(data) > 0:
            summary = {
                "symbol": symbol.upper(),
                "period_type": period,
                "periods_retrieved": len(data),
                "latest_period": data[0].get("date", "N/A") if data else "N/A",
                "key_ratios_latest": {
                    # Liquidity Ratios
                    "current_ratio": data[0].get("currentRatio", "N/A"),
                    "quick_ratio": data[0].get("quickRatio", "N/A"),
                    "cash_ratio": data[0].get("cashRatio", "N/A"),
                    # Profitability Ratios
                    "gross_profit_margin": data[0].get("grossProfitMargin", "N/A"),
                    "operating_profit_margin": data[0].get("operatingProfitMargin", "N/A"),
                    "net_profit_margin": data[0].get("netProfitMargin", "N/A"),
                    "return_on_assets": data[0].get("returnOnAssets", "N/A"),
                    "return_on_equity": data[0].get("returnOnEquity", "N/A"),
                    # Leverage Ratios
                    "debt_ratio": data[0].get("debtRatio", "N/A"),
                    "debt_to_equity": data[0].get("debtEquityRatio", "N/A"),
                    "times_interest_earned": data[0].get("timesInterestEarned", "N/A"),
                    # Efficiency Ratios
                    "asset_turnover": data[0].get("assetTurnover", "N/A"),
                    "inventory_turnover": data[0].get("inventoryTurnover", "N/A"),
                    "receivables_turnover": data[0].get("receivablesTurnover", "N/A")
                } if data else {},
                "full_ratios": data
            }
            
            return json.dumps(summary, indent=2)
        else:
            return json.dumps(data, indent=2)
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Error fetching financial ratios for {symbol}: {e}"
        print(error_msg)
        return error_msg
    except json.JSONDecodeError as e:
        error_msg = f"Error parsing financial ratios response for {symbol}: {e}"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Unexpected error in financial_ratios_tool for {symbol}: {e}"
        print(error_msg)
        return error_msg


# Additional utility function to test API connectivity
def test_fmp_api_connection(tool_context: ToolContext = None) -> str:
    """
    Tests the Financial Modeling Prep API connection.
    
    Returns:
        Status message indicating whether the API is accessible.
    """
    print("--- TOOL: test_fmp_api_connection called ---")
    
    try:
        # Test with a simple company profile request for Apple
        url = f"{FMP_BASE_URL}/profile"
        params = {
            "symbol": "AAPL",
            "apikey": FMP_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data and isinstance(data, list) and len(data) > 0:
            return "Financial Modeling Prep API connection successful"
        else:
            return "Financial Modeling Prep API connection established but no data returned"
            
    except requests.exceptions.RequestException as e:
        return f"Financial Modeling Prep API connection failed: {e}"
    except Exception as e:
        return f"Unexpected error testing FMP API: {e}"