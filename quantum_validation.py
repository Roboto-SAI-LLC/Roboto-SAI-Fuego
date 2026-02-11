#!/usr/bin/env python3
"""
Quantum Engine Validation Script
Validates imports and initialization for Qiskit, Qiskit-Aer, QuTiP
"""

import sys
import traceback

def test_import(module_name, description):
    try:
        __import__(module_name)
        print(f"{description}: SUCCESS")
        return True
    except ImportError as e:
        print(f"{description}: FAILED - {str(e)}")
        return False
    except Exception as e:
        print(f"{description}: ERROR - {str(e)}")
        return False

def main():
    print("=== Quantum Engine Validation Protocol ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python prefix: {sys.prefix}")
    print()

    # Test imports
    qiskit_ok = test_import("qiskit", "qiskit import")
    qiskit_aer_ok = test_import("qiskit_aer", "qiskit-aer import")
    qutip_ok = test_import("qutip", "qutip import")

    print()

    # Test quantum engine
    try:
        from backend.services.quantum_engine import QuantumEngine
        print("QuantumEngine import: SUCCESS")

        # Initialize
        engine = QuantumEngine()
        status = engine.get_status()
        print(f"Engine initialization: SUCCESS")
        print(f"Session ID: {status.get('session_id', 'N/A')}")
        print(f"Status: {status.get('status', 'N/A')}")

        # Check capabilities
        quantum_cap = status.get('quantum', {})
        print(f"Qiskit available: {quantum_cap.get('qiskit', False)}")
        print(f"QuTiP available: {quantum_cap.get('qutip', False)}")
        print(f"Hybrid bridge: {quantum_cap.get('hybrid_bridge', False)}")
        print(f"Kernel initialized: {quantum_cap.get('kernel_initialized', False)}")

        final_readiness = all([
            quantum_cap.get('qiskit', False),
            quantum_cap.get('qutip', False),
            quantum_cap.get('kernel_initialized', False)
        ])
        print(f"Final readiness state: {'READY' if final_readiness else 'PARTIAL'}")

        # Validation report
        validation_report = {
            "qiskit_import_status": qiskit_ok,
            "qiskit_aer_backend_availability": qiskit_aer_ok,
            "qutip_import_status": qutip_ok,
            "hybrid_bridge_readiness": quantum_cap.get('hybrid_bridge', False),
            "updated_session_id": status.get('session_id', 'N/A'),
            "final_readiness_state": 'READY' if final_readiness else 'PARTIAL'
        }

        print()
        print("=== Validation Report ===")
        for key, value in validation_report.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"QuantumEngine initialization: FAILED - {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()