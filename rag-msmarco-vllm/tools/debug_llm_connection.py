import asyncio
import sys
from pathlib import Path

# Add app directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.llm import get_production_llm, validate_llm_connection


async def main():
    """
    Standalone async script to test the LLM connection outside of FastAPI.
    """
    print("--- Independent LLM Connection Test ---")
    print("Attempting to initialize LLM client...")
    
    try:
        # Get the same LLM instance the app would use
        llm = get_production_llm()
        print("LLM client initialized successfully.")
        
        print("\nAttempting to invoke LLM with a test prompt...")
        # Run the synchronous validation function in an async executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            validate_llm_connection, 
            llm, 
            "Hello, this is a test."
        )
        
        print("\n--- Test Result ---")
        print(f"Status: {result.get('status')}")
        if result.get('status') == 'success':
            print(f"Response Preview: {result.get('response_preview')}")
        else:
            print(f"Error: {result.get('error')}")
            print(f"Error Type: {result.get('error_type')}")
        print("--------------------")

    except Exception as e:
        print(f"\n--- An unexpected error occurred ---")
        print(f"Error: {e}")
        print(f"Error Type: {type(e).__name__}")
        print("------------------------------------")

if __name__ == "__main__":
    asyncio.run(main())
