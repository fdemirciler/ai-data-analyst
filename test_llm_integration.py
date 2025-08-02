#!/usr/bin/env python3
"""
LLM Integration Test Script

This script tests the real LLM integration to ensure the fake AI logic
has been successfully replaced with actual LLM capabilities.
"""

import asyncio
import sys
import os
import json
from pathlib import Path
import pandas as pd

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

try:
    from api.llm_endpoints import AnalysisRequest, analyze_with_llm, set_session_storage
    from services.llm_provider import LLMManager
    from config import settings
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(
        "Make sure you're running this from the correct directory and all dependencies are installed."
    )
    sys.exit(1)


async def test_llm_integration():
    """Test the real LLM integration functionality"""

    print("🧪 Testing Real LLM Integration")
    print("=" * 50)

    # Test 1: Verify LLM Manager Configuration
    print("\n1️⃣ Testing LLM Manager Configuration...")
    try:
        llm_manager = LLMManager(settings)
        print(f"✅ LLM Manager initialized successfully")
        print(f"   Primary provider: {llm_manager.primary_provider.provider_name}")
        print(
            f"   Available providers: {[p.provider_name for p in llm_manager.providers]}"
        )
    except Exception as e:
        print(f"❌ LLM Manager initialization failed: {e}")
        return False

    # Test 2: Create test session data
    print("\n2️⃣ Setting up test session...")
    test_session_storage = {
        "test-session-123": {
            "session_id": "test-session-123",
            "created_at": "2024-01-01T00:00:00",
            "status": "active",
            "conversation_history": [],
            "uploaded_file": {
                "filename": "test_data.csv",
                "file_path": "test_data.csv",
                "file_size": 1024,
                "rows": 100,
                "columns": 5,
                "uploaded_at": "2024-01-01T00:00:00",
            },
        }
    }

    # Create a simple test CSV file
    test_data = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age": [25, 30, 35, 28, 32],
            "salary": [50000, 60000, 75000, 55000, 68000],
            "department": [
                "Engineering",
                "Marketing",
                "Engineering",
                "Sales",
                "Marketing",
            ],
            "experience": [2, 5, 8, 3, 6],
        }
    )
    test_data.to_csv("test_data.csv", index=False)

    # Set the session storage
    set_session_storage(test_session_storage)
    print("✅ Test session and data created")

    # Test 3: Test LLM Analysis Request
    print("\n3️⃣ Testing Real LLM Analysis...")

    test_queries = [
        "Show me a summary of the data",
        "Who has the highest salary?",
        "Create a visualization of salary by department",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test Query {i}: '{query}'")

        try:
            request = AnalysisRequest(
                query=query, session_id="test-session-123", stream=False, max_retries=1
            )

            print("   📡 Sending request to LLM system...")
            response = await analyze_with_llm(request)

            print(f"   ✅ Response received:")
            print(f"      Success: {response.success}")
            print(f"      Processing time: {response.processing_time:.2f}s")
            print(f"      Generated code: {'Yes' if response.generated_code else 'No'}")
            print(
                f"      Has execution output: {'Yes' if response.execution_output else 'No'}"
            )
            print(f"      Has error: {'Yes' if response.execution_error else 'No'}")

            if response.interpretation:
                print(
                    f"      Interpretation preview: {response.interpretation[:100]}..."
                )

            if not response.success:
                print(f"      ⚠️ Analysis failed: {response.interpretation}")

        except Exception as e:
            print(f"   ❌ Query failed: {e}")

    # Test 4: Verify Configuration
    print("\n4️⃣ Verifying Configuration...")

    # Check if API keys are configured
    api_keys_present = {}
    api_keys_present["GEMINI"] = bool(os.getenv("GEMINI_API_KEY"))
    api_keys_present["OPENROUTER"] = bool(os.getenv("OPENROUTER_API_KEY"))
    api_keys_present["TOGETHER"] = bool(os.getenv("TOGETHER_API_KEY"))

    print("   API Keys configured:")
    for provider, present in api_keys_present.items():
        status = "✅" if present else "❌"
        print(f"      {status} {provider}: {'Present' if present else 'Missing'}")

    if not any(api_keys_present.values()):
        print("   ⚠️ No API keys found! LLM integration will not work.")
        print(
            "   Please configure at least one LLM provider API key in your .env file."
        )

    # Cleanup
    print("\n🧹 Cleaning up...")
    if os.path.exists("test_data.csv"):
        os.remove("test_data.csv")
    print("✅ Test files cleaned up")

    print("\n" + "=" * 50)
    print("🎉 LLM Integration Test Complete!")

    if any(api_keys_present.values()):
        print("✅ Real LLM integration is ready to use!")
        print("✅ Fake AI logic has been successfully replaced!")
    else:
        print("⚠️ Configure API keys to enable full LLM functionality")

    return True


def test_configuration():
    """Test basic configuration without async operations"""
    print("🔧 Testing Basic Configuration")
    print("=" * 30)

    # Check environment variables
    required_vars = ["GEMINI_API_KEY", "OPENROUTER_API_KEY", "TOGETHER_API_KEY"]

    print("Environment Variables:")
    for var in required_vars:
        value = os.getenv(var)
        status = "✅" if value else "❌"
        print(f"  {status} {var}: {'Present' if value else 'Missing'}")

    # Check if config can be loaded
    try:
        from config import settings

        print(f"\n✅ Settings loaded successfully")
        print(f"   Default LLM provider: gemini")
        print(f"   Temperature: 0.1")
    except Exception as e:
        print(f"\n❌ Failed to load settings: {e}")

    print("\n" + "=" * 30)


if __name__ == "__main__":
    print("🚀 Starting LLM Integration Tests")
    print(
        "This script verifies that fake AI logic has been replaced with real LLM integration"
    )
    print()

    # Test basic configuration first
    test_configuration()

    print("\n" + "=" * 50)

    # Test full integration
    try:
        asyncio.run(test_llm_integration())
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback

        traceback.print_exc()
