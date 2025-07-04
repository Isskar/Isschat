# Weaviate gRPC Timeout Fix for Azure Deployment

## Problem
The application experiences intermittent gRPC timeout errors when connecting to Weaviate Cloud from Azure:
```
Query call with protocol GRPC search failed with message Deadline Exceeded
```

This error occurs approximately 1 out of 5 times on the Azure deployment but not locally.

## Root Cause
- Weaviate Python client v4 uses gRPC by default for better performance
- Network latency between Azure and Weaviate Cloud can cause gRPC timeouts
- The default timeout settings may be too aggressive for cross-cloud connections

## Solution Implemented

### 1. Extended Timeout Configuration
- Increased connection timeout to 30 seconds
- Increased query timeout to 60 seconds  
- Increased insert timeout to 60 seconds

### 2. Retry Logic with Exponential Backoff
- Added automatic retry for connection initialization (3 attempts)
- Added automatic retry for search operations (3 attempts)
- Implements exponential backoff (2s, 4s, 8s delays)

### 3. Custom Headers for Azure
- Added `X-Azure-Source: true` header to identify Azure requests
- Added `X-Timeout-Seconds: 60` header as a timeout hint

### 4. REST-only Fallback Option
As a last resort, you can force the use of Weaviate v3 client which only uses REST protocol:

```bash
# Set this environment variable in Azure App Service
WEAVIATE_USE_V3_CLIENT=true
```

## Configuration Steps for Azure

### Option 1: Use Enhanced v4 Client (Recommended)
No additional configuration needed. The improved timeout and retry logic should handle most cases.

### Option 2: Force REST-only Protocol (If issues persist)
1. Go to Azure Portal > Your App Service > Configuration
2. Add new Application Setting:
   - Name: `WEAVIATE_USE_V3_CLIENT`
   - Value: `true`
3. Save and restart the app

## Monitoring
The application will log:
- Connection attempts and successes
- Retry attempts with detailed error messages
- Which client version (v3 or v4) is being used

Look for these log messages:
- `"Weaviate v4 client connected successfully (attempt X)"`
- `"Weaviate connection timeout (attempt X/3): ... Retrying in Xs..."`
- `"Using Weaviate v3 client (REST-only) as configured"`

## Performance Considerations
- v4 client with gRPC is faster but may have connectivity issues
- v3 client with REST is more reliable but slightly slower
- The retry logic adds resilience but may increase response time for failed attempts

## Next Steps if Issues Persist
1. Contact Weaviate support about gRPC connectivity from Azure regions
2. Consider using a Weaviate instance in the same cloud provider (Azure)
3. Implement a circuit breaker pattern for better fault tolerance