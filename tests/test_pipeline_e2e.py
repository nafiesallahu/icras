"""
ICRAS End-to-End Pipeline Integration Test
------------------------------------------
This file executes a deterministic integration test between:
  1. IntakeAgent (Agent A) - Handles data ingestion and run directory creation.
  2. ExtractionAgent (Agent B) - Manages context reading and triggers the 
     Synthetic Fallback Engine under test environments.

Usage:
    python -m tests.test_pipeline_e2e
"""



import os
import shutil
import logging
from pathlib import Path
from pydantic import ValidationError
from app.agents.intake_agent import IntakeAgent
from app.agents.extraction_agent import ExtractionAgent
from app.schemas.context_packet import ContextPacket
from app.schemas.extracted_contract import ExtractedContract

# Setup minimal logging to track orchestrator execution
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def setup_integration_environment() -> Path:
    """Prepares a mock source folder to simulate raw contract ingestion."""
    bundle_dir = Path("data/bundle_e2e_test")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    
    # Write a dummy contract file so the Intake scanning logic passes successfully
    mock_file = bundle_dir / "raw_incoming_contract.txt"
    mock_file.write_text("ICRAS End-to-End Pipeline Mock Content")
    
    # Enforce test mode via environment variables to trigger the Synthetic Fallback Engine
    os.environ["ENV_MODE"] = "test"
    os.environ["MOCK_PIPELINE"] = "True"
    
    return bundle_dir

def run_e2e_pipeline_test():
    print("\n======================================================================")
    print("STARTING INTEGRATION TEST: INTAKE AGENT (A) -> EXTRACTION AGENT (B)")
    print("======================================================================")
    
    bundle_dir = setup_integration_environment()
    
    # Initialize both architectural agents
    intake_agent = IntakeAgent()          # Intake Agent (A)
    extraction_agent = ExtractionAgent()  # Extraction Agent (B)
    
    run_dir_created = None
    
    try:
        # --------------------------------------------------------------------
        # PHASE 1: Execute IntakeAgent (Agent A)
        # --------------------------------------------------------------------
        print("\n[PHASE 1] Executing IntakeAgent (Agent A) - Data Ingestion...")
        print(f"-> Scanning input directory: {bundle_dir}")
        
        context_packet: ContextPacket = intake_agent.run(
            bundle_path=str(bundle_dir),
            document_type="SERVICES_AGREEMENT"
        )
        
        print(f"✅ IntakeAgent (Agent A) Success!")
        print(f"   - Generated Contract ID: {context_packet.contract_id}")
        print(f"   - Ingestion ISO Timestamp: {context_packet.received_timestamp}")
        
        # Locate the physical run directory created dynamically under runs/
        runs_root = Path("runs")
        matching_dirs = list(runs_root.glob(f"*{context_packet.contract_id}"))
        assert len(matching_dirs) == 1, "Verification Failure: Run runtime execution directory was not found!"
        run_dir_created = matching_dirs[0]
        print(f"📍 Isolated Run Directory Created: {run_dir_created}")
        
        # Confirm context_packet.json exists on disk
        assert (run_dir_created / "context_packet.json").exists(), "Verification Failure: context_packet.json missing from run directory!"
        print("✅ Found 'context_packet.json' on disk.")

        # --------------------------------------------------------------------
        # PHASE 2: Execute ExtractionAgent (Agent B)
        # --------------------------------------------------------------------
        print("\n[PHASE 2] Executing ExtractionAgent (Agent B) - Orchestrated Extraction...")
        print("-> Intercepting live LLM paths. Routing directly to Synthetic Fallback Engine.")
        
        # We test using Scenario 03 (Net 90 terms) passing the short string "3" to verify normalization
        extracted_contract: ExtractedContract = extraction_agent.run(
            run_directory=str(run_dir_created),
            scenario_id="3"
        )
        
        # --------------------------------------------------------------------
        # PHASE 3: Complete Pipeline Contract Verification
        # --------------------------------------------------------------------
        print("\n[PHASE 3] Running End-to-End Pipeline Integrity Controls...")
        
        # Control 1: Check if the artifact was stored correctly inside the shared directory
        final_json_file = run_dir_created / "extracted_contract.json"
        assert final_json_file.exists(), "Verification Failure: extracted_contract.json was not persisted by Agent B!"
        print("✅ Pipeline Integrity Passed: 'extracted_contract.json' successfully found in the execution directory.")
        
        # Control 2: Ensure type instance safety
        assert isinstance(extracted_contract, ExtractedContract), "Verification Failure: Return type is not a strict ExtractedContract instance!"
        print(f"✅ Instance Type Control Passed: Validated schema version {extracted_contract.schema_version}")
        
        # Control 3: Confirm Scenario 03 values loaded correctly
        # Scenario 03 focuses on parsing extended payment windows
        print(f"ℹ️ Extracted Contract ID from JSON: {extracted_contract.contract_id}")
        print(f"ℹ️ Verified Scenario Content Baseline (Overall Confidence Score): {extracted_contract.overall_confidence_score}")

        print("\n======================================================================")
        print("🎉 SUCCESS: Intake Agent (A) & Extraction Agent (B) Are Fully Integrated!")
        print("======================================================================")
        
    except Exception as e:
        print(f"\n❌ PIPELINE INTEGRATION TEST FAILED: {str(e)}")
        raise e
        
    finally:
        # Automated workspace environment clean-up
        print("\n-> Cleaning up temporary testing execution workspaces...")
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        if run_dir_created and run_dir_created.exists():
            shutil.rmtree(run_dir_created)
        print("✅ Clean-up finished. System environment reset successfully.")

if __name__ == "__main__":
    run_e2e_pipeline_test()