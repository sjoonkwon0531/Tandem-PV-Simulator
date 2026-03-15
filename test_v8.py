"""
Quick test for V8 modules
"""

import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / 'utils'))

def test_imports():
    """Test all V8 module imports"""
    print("Testing V8 module imports...")
    
    try:
        from model_zoo import ModelRegistry, ModelCard
        print("✅ model_zoo imported")
    except Exception as e:
        print(f"❌ model_zoo failed: {e}")
        return False
    
    try:
        from api_generator import APISpecGenerator, RateLimiter, UsageTracker
        print("✅ api_generator imported")
    except Exception as e:
        print(f"❌ api_generator failed: {e}")
        return False
    
    try:
        from benchmarks import BenchmarkSuite, StatisticalTests, ReproducibilityReport
        print("✅ benchmarks imported")
    except Exception as e:
        print(f"❌ benchmarks failed: {e}")
        return False
    
    try:
        from education import TutorialLibrary, Glossary, QuizEngine, GuidedWorkflow
        print("✅ education imported")
    except Exception as e:
        print(f"❌ education failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of V8 modules"""
    print("\nTesting basic functionality...")
    
    from model_zoo import ModelRegistry
    from api_generator import APISpecGenerator
    from benchmarks import BenchmarkSuite
    from education import Glossary
    
    # Test model registry
    try:
        registry = ModelRegistry()
        print("✅ ModelRegistry created")
    except Exception as e:
        print(f"❌ ModelRegistry failed: {e}")
        return False
    
    # Test API spec generator
    try:
        api_gen = APISpecGenerator()
        spec = api_gen.generate_spec()
        assert 'openapi' in spec
        assert spec['openapi'] == '3.0.0'
        print("✅ APISpecGenerator works")
    except Exception as e:
        print(f"❌ APISpecGenerator failed: {e}")
        return False
    
    # Test benchmark suite
    try:
        suite = BenchmarkSuite()
        assert len(suite.benchmarks) == 3
        print("✅ BenchmarkSuite works")
    except Exception as e:
        print(f"❌ BenchmarkSuite failed: {e}")
        return False
    
    # Test glossary
    try:
        definition = Glossary.get_definition('Bandgap (Eg)')
        assert 'energy' in definition.lower()
        print("✅ Glossary works")
    except Exception as e:
        print(f"❌ Glossary failed: {e}")
        return False
    
    return True

if __name__ == '__main__':
    print("="*60)
    print("V8 Module Tests")
    print("="*60)
    
    imports_ok = test_imports()
    
    if imports_ok:
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n" + "="*60)
            print("✅ ALL TESTS PASSED")
            print("="*60)
            sys.exit(0)
        else:
            print("\n" + "="*60)
            print("❌ FUNCTIONALITY TESTS FAILED")
            print("="*60)
            sys.exit(1)
    else:
        print("\n" + "="*60)
        print("❌ IMPORT TESTS FAILED")
        print("="*60)
        sys.exit(1)
