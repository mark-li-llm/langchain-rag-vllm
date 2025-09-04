#!/usr/bin/env python3
"""Load testing tool for RAG API endpoints.

This script generates concurrent requests to test the API performance
and identify bottlenecks under load.
"""

import sys
import os
import asyncio
import aiohttp
import json
import time
import random
import argparse
from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from statistics import mean, median

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class RequestResult:
    """Result of a single API request."""
    query: str
    status_code: int
    response_time_ms: float
    success: bool
    error: str = ""
    response_size: int = 0
    trace_id: str = ""


@dataclass
class LoadTestConfig:
    """Configuration for load test."""
    base_url: str
    auth_token: str
    queries: List[str]
    concurrent_users: int
    duration_seconds: int
    requests_per_user: int
    endpoint: str = "/v1/query"


class LoadTester:
    """Handles load testing of the RAG API."""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results: List[RequestResult] = []
        self.start_time = 0
        self.end_time = 0
    
    async def make_request(self, session: aiohttp.ClientSession, query: str) -> RequestResult:
        """Make a single API request."""
        url = f"{self.config.base_url}{self.config.endpoint}"
        headers = {
            "Authorization": f"Bearer {self.config.auth_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "query": query,
            "stream": False,  # Non-streaming for easier measurement
            "top_k": 5
        }
        
        start_time = time.time()
        
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    response_data = await response.json()
                    response_size = len(json.dumps(response_data))
                    trace_id = response_data.get("trace_id", "")
                    
                    return RequestResult(
                        query=query,
                        status_code=response.status,
                        response_time_ms=response_time,
                        success=True,
                        response_size=response_size,
                        trace_id=trace_id
                    )
                else:
                    error_text = await response.text()
                    return RequestResult(
                        query=query,
                        status_code=response.status,
                        response_time_ms=response_time,
                        success=False,
                        error=error_text[:200]  # Truncate long errors
                    )
                    
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return RequestResult(
                query=query,
                status_code=0,
                response_time_ms=response_time,
                success=False,
                error=str(e)[:200]
            )
    
    async def user_session(self, user_id: int, session: aiohttp.ClientSession):
        """Simulate a single user making requests."""
        user_results = []
        
        for i in range(self.config.requests_per_user):
            # Select random query
            query = random.choice(self.config.queries)
            
            # Make request
            result = await self.make_request(session, query)
            user_results.append(result)
            
            # Small delay between requests from same user
            await asyncio.sleep(random.uniform(0.1, 0.5))
        
        return user_results
    
    async def run_test(self) -> Dict[str, Any]:
        """Run the complete load test."""
        print(f"Starting load test...")
        print(f"- Concurrent users: {self.config.concurrent_users}")
        print(f"- Requests per user: {self.config.requests_per_user}")
        print(f"- Total requests: {self.config.concurrent_users * self.config.requests_per_user}")
        print(f"- Queries pool: {len(self.config.queries)}")
        print("")
        
        self.start_time = time.time()
        
        # Configure session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=self.config.concurrent_users * 2,
            limit_per_host=self.config.concurrent_users * 2,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=60)  # 60 second timeout
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create tasks for all users
            tasks = [
                self.user_session(user_id, session)
                for user_id in range(self.config.concurrent_users)
            ]
            
            # Wait for all users to complete
            user_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        self.end_time = time.time()
        
        # Flatten results
        for user_result in user_results:
            if isinstance(user_result, Exception):
                print(f"User session failed: {user_result}")
                continue
            self.results.extend(user_result)
        
        return self.analyze_results()
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze test results and return summary."""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Basic stats
        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r.success)
        failed_requests = total_requests - successful_requests
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        
        # Response time stats (only successful requests)
        successful_times = [r.response_time_ms for r in self.results if r.success]
        
        if successful_times:
            avg_response_time = mean(successful_times)
            median_response_time = median(successful_times)
            min_response_time = min(successful_times)
            max_response_time = max(successful_times)
            
            # Percentiles
            sorted_times = sorted(successful_times)
            p95_index = int(0.95 * len(sorted_times))
            p99_index = int(0.99 * len(sorted_times))
            p95_response_time = sorted_times[p95_index] if p95_index < len(sorted_times) else max_response_time
            p99_response_time = sorted_times[p99_index] if p99_index < len(sorted_times) else max_response_time
        else:
            avg_response_time = median_response_time = min_response_time = max_response_time = 0
            p95_response_time = p99_response_time = 0
        
        # Throughput
        test_duration = self.end_time - self.start_time
        requests_per_second = total_requests / test_duration if test_duration > 0 else 0
        successful_rps = successful_requests / test_duration if test_duration > 0 else 0
        
        # Error analysis
        error_counts = {}
        status_counts = {}
        
        for result in self.results:
            status_counts[result.status_code] = status_counts.get(result.status_code, 0) + 1
            
            if not result.success and result.error:
                error_type = result.error.split(':')[0]  # Get error type
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Response size stats
        sizes = [r.response_size for r in self.results if r.success and r.response_size > 0]
        avg_response_size = mean(sizes) if sizes else 0
        
        return {
            "test_config": {
                "concurrent_users": self.config.concurrent_users,
                "requests_per_user": self.config.requests_per_user,
                "duration_seconds": test_duration,
                "queries_count": len(self.config.queries)
            },
            "summary": {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": success_rate
            },
            "performance": {
                "requests_per_second": requests_per_second,
                "successful_rps": successful_rps,
                "avg_response_time_ms": avg_response_time,
                "median_response_time_ms": median_response_time,
                "min_response_time_ms": min_response_time,
                "max_response_time_ms": max_response_time,
                "p95_response_time_ms": p95_response_time,
                "p99_response_time_ms": p99_response_time,
                "avg_response_size_bytes": avg_response_size
            },
            "errors": {
                "status_codes": status_counts,
                "error_types": error_counts
            }
        }


def load_test_queries(file_path: str = None) -> List[str]:
    """Load test queries from file or return default set."""
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        return queries
    
    # Default test queries
    return [
        "What is machine learning?",
        "How does artificial intelligence work?",
        "Explain neural networks",
        "What are the applications of deep learning?",
        "Define natural language processing",
        "How do recommendation systems work?",
        "What is computer vision?",
        "Explain reinforcement learning",
        "What are decision trees?",
        "How does gradient descent work?",
        "What is overfitting in machine learning?",
        "Explain feature engineering",
        "What are support vector machines?",
        "How do convolutional neural networks work?",
        "What is transfer learning?",
        "Explain ensemble methods",
        "What is clustering in machine learning?",
        "How does cross-validation work?",
        "What are hyperparameters?",
        "Explain dimensionality reduction"
    ]


def print_results(results: Dict[str, Any]):
    """Print formatted test results."""
    print("\nLoad Test Results")
    print("=" * 50)
    
    # Test configuration
    config = results["test_config"]
    print(f"Configuration:")
    print(f"  - Concurrent users: {config['concurrent_users']}")
    print(f"  - Requests per user: {config['requests_per_user']}")
    print(f"  - Test duration: {config['duration_seconds']:.2f}s")
    print(f"  - Query pool size: {config['queries_count']}")
    print("")
    
    # Summary
    summary = results["summary"]
    print(f"Summary:")
    print(f"  - Total requests: {summary['total_requests']}")
    print(f"  - Successful: {summary['successful_requests']}")
    print(f"  - Failed: {summary['failed_requests']}")
    print(f"  - Success rate: {summary['success_rate']:.2%}")
    print("")
    
    # Performance
    perf = results["performance"]
    print(f"Performance:")
    print(f"  - Requests/second: {perf['requests_per_second']:.2f}")
    print(f"  - Successful RPS: {perf['successful_rps']:.2f}")
    print(f"  - Avg response time: {perf['avg_response_time_ms']:.2f}ms")
    print(f"  - Median response time: {perf['median_response_time_ms']:.2f}ms")
    print(f"  - P95 response time: {perf['p95_response_time_ms']:.2f}ms")
    print(f"  - P99 response time: {perf['p99_response_time_ms']:.2f}ms")
    print(f"  - Min response time: {perf['min_response_time_ms']:.2f}ms")
    print(f"  - Max response time: {perf['max_response_time_ms']:.2f}ms")
    print(f"  - Avg response size: {perf['avg_response_size_bytes']:.0f} bytes")
    print("")
    
    # Error analysis
    errors = results["errors"]
    if errors["status_codes"]:
        print(f"Status codes:")
        for code, count in sorted(errors["status_codes"].items()):
            print(f"  - {code}: {count}")
        print("")
    
    if errors["error_types"]:
        print(f"Error types:")
        for error_type, count in sorted(errors["error_types"].items()):
            print(f"  - {error_type}: {count}")
        print("")


async def main():
    """Main function for load testing."""
    parser = argparse.ArgumentParser(description="Load test RAG API")
    
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="Base URL of the API"
    )
    parser.add_argument(
        "--token",
        required=True,
        help="Bearer token for authentication"
    )
    parser.add_argument(
        "--users",
        type=int,
        default=5,
        help="Number of concurrent users"
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=10,
        help="Number of requests per user"
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        help="File containing test queries (one per line)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--endpoint",
        default="/v1/query",
        help="API endpoint to test"
    )
    
    args = parser.parse_args()
    
    # Load test queries
    queries = load_test_queries(args.queries_file)
    print(f"Loaded {len(queries)} test queries")
    
    # Create configuration
    config = LoadTestConfig(
        base_url=args.url,
        auth_token=args.token,
        queries=queries,
        concurrent_users=args.users,
        duration_seconds=0,  # Will be calculated
        requests_per_user=args.requests,
        endpoint=args.endpoint
    )
    
    # Run load test
    tester = LoadTester(config)
    results = await tester.run_test()
    
    # Print results
    print_results(results)
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    
    # Return appropriate exit code
    success_rate = results.get("summary", {}).get("success_rate", 0)
    return success_rate > 0.95  # Consider test successful if >95% success rate


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nLoad test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"Load test failed: {e}")
        sys.exit(1)