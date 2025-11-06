# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Security Test Suite for CVE-001: Insecure Deserialization

This test suite validates the presence of insecure deserialization vulnerabilities
in the transformers codebase, specifically:
1. Unsafe usage of pickle.load()
2. Unsafe usage of yaml.load() with BaseLoader/FullLoader

These tests are designed to DETECT vulnerabilities, not fix them.
"""

import ast
import os
import re
import unittest
from pathlib import Path


class TestCVE001InsecureDeserialization(unittest.TestCase):
    """Test suite for detecting insecure deserialization patterns (CVE-001)"""

    def setUp(self):
        """Set up test environment"""
        self.repo_root = Path(__file__).parent.parent.parent
        self.vulnerable_files = []
        self.pickle_load_instances = []
        self.unsafe_yaml_load_instances = []

    def test_01_detect_pickle_load_in_src(self):
        """
        Test 1: Detect pickle.load usage in src/transformers directory
        
        VULNERABILITY: pickle.load can execute arbitrary code during deserialization.
        This test verifies that pickle.load is present in the source code, which
        poses a CRITICAL security risk.
        """
        src_dir = self.repo_root / "src" / "transformers"
        pickle_pattern = re.compile(r'\bpickle\.load\(')
        
        found_instances = []
        
        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for line_num, line in enumerate(content.split('\n'), 1):
                        if pickle_pattern.search(line):
                            found_instances.append({
                                'file': str(py_file.relative_to(self.repo_root)),
                                'line': line_num,
                                'content': line.strip()
                            })
            except Exception:
                continue
        
        self.pickle_load_instances.extend(found_instances)
        
        # This test PASSES if vulnerabilities are found (we're testing for presence)
        self.assertGreater(
            len(found_instances), 
            0, 
            "EXPECTED: pickle.load vulnerabilities should be present in src/transformers. "
            "This confirms the vulnerability exists."
        )
        
        print(f"\n[VULNERABILITY DETECTED] Found {len(found_instances)} pickle.load instances in src/transformers:")
        for instance in found_instances:
            print(f"  - {instance['file']}:{instance['line']} -> {instance['content']}")

    def test_02_detect_pickle_load_in_examples(self):
        """
        Test 2: Detect pickle.load usage in examples directory
        
        VULNERABILITY: Examples containing pickle.load can mislead users into
        using unsafe deserialization patterns.
        """
        examples_dir = self.repo_root / "examples"
        if not examples_dir.exists():
            self.skipTest("Examples directory not found")
            
        pickle_pattern = re.compile(r'\bpickle\.load\(')
        
        found_instances = []
        
        for py_file in examples_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for line_num, line in enumerate(content.split('\n'), 1):
                        if pickle_pattern.search(line):
                            found_instances.append({
                                'file': str(py_file.relative_to(self.repo_root)),
                                'line': line_num,
                                'content': line.strip()
                            })
            except Exception:
                continue
        
        self.pickle_load_instances.extend(found_instances)
        
        # This test PASSES if vulnerabilities are found
        self.assertGreater(
            len(found_instances), 
            0, 
            "EXPECTED: pickle.load vulnerabilities should be present in examples/. "
            "This confirms the vulnerability exists."
        )
        
        print(f"\n[VULNERABILITY DETECTED] Found {len(found_instances)} pickle.load instances in examples/:")
        for instance in found_instances:
            print(f"  - {instance['file']}:{instance['line']} -> {instance['content']}")

    def test_03_detect_unsafe_yaml_load(self):
        """
        Test 3: Detect unsafe yaml.load usage with BaseLoader or FullLoader
        
        VULNERABILITY: yaml.load with BaseLoader/FullLoader can execute arbitrary
        Python code. yaml.safe_load should be used instead.
        """
        src_dir = self.repo_root / "src" / "transformers"
        
        # Pattern to detect yaml.load with unsafe loaders
        unsafe_yaml_pattern = re.compile(
            r'yaml\.load\([^)]*Loader\s*=\s*yaml\.(BaseLoader|FullLoader|Loader)\s*\)'
        )
        
        found_instances = []
        
        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for line_num, line in enumerate(content.split('\n'), 1):
                        if unsafe_yaml_pattern.search(line):
                            found_instances.append({
                                'file': str(py_file.relative_to(self.repo_root)),
                                'line': line_num,
                                'content': line.strip()
                            })
            except Exception:
                continue
        
        self.unsafe_yaml_load_instances.extend(found_instances)
        
        # This test PASSES if vulnerabilities are found
        self.assertGreater(
            len(found_instances), 
            0, 
            "EXPECTED: unsafe yaml.load vulnerabilities should be present. "
            "This confirms the vulnerability exists."
        )
        
        print(f"\n[VULNERABILITY DETECTED] Found {len(found_instances)} unsafe yaml.load instances:")
        for instance in found_instances:
            print(f"  - {instance['file']}:{instance['line']} -> {instance['content']}")

    def test_04_verify_pickle_import_presence(self):
        """
        Test 4: Verify that pickle module is imported in vulnerable files
        
        This test confirms that the pickle module is actually being imported,
        which is a prerequisite for the vulnerability to exist.
        """
        src_dir = self.repo_root / "src" / "transformers"
        pickle_import_pattern = re.compile(r'^\s*import\s+pickle|^\s*from\s+pickle\s+import', re.MULTILINE)
        
        found_files = []
        
        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if pickle_import_pattern.search(content):
                        found_files.append(str(py_file.relative_to(self.repo_root)))
            except Exception:
                continue
        
        # This test PASSES if pickle imports are found
        self.assertGreater(
            len(found_files), 
            0, 
            "EXPECTED: pickle imports should be present in src/transformers. "
            "This confirms potential for vulnerability."
        )
        
        print(f"\n[VULNERABILITY DETECTED] Found {len(found_files)} files importing pickle:")
        for file in found_files[:10]:  # Show first 10
            print(f"  - {file}")

    def test_05_check_rag_retrieval_vulnerability(self):
        """
        Test 5: Specific check for RAG retrieval module vulnerability
        
        The RAG retrieval module contains multiple pickle.load instances for loading
        passages and index metadata, which is a high-risk area.
        """
        rag_file = self.repo_root / "src" / "transformers" / "models" / "rag" / "retrieval_rag.py"
        
        if not rag_file.exists():
            self.skipTest("RAG retrieval file not found")
        
        with open(rag_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        pickle_loads = len(re.findall(r'\bpickle\.load\(', content))
        
        # This test PASSES if vulnerabilities are found
        self.assertGreater(
            pickle_loads,
            0,
            "EXPECTED: RAG retrieval module should contain pickle.load vulnerabilities. "
            "This is a HIGH-RISK area as it loads external data."
        )
        
        print(f"\n[HIGH-RISK VULNERABILITY] RAG retrieval module contains {pickle_loads} pickle.load calls")

    def test_06_check_language_modeling_dataset_vulnerability(self):
        """
        Test 6: Check language modeling datasets for pickle.load
        
        Data loading utilities that use pickle.load are particularly dangerous as
        they process external data sources.
        """
        dataset_file = self.repo_root / "src" / "transformers" / "data" / "datasets" / "language_modeling.py"
        
        if not dataset_file.exists():
            self.skipTest("Language modeling dataset file not found")
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        pickle_loads = len(re.findall(r'\bpickle\.load\(', content))
        
        # This test PASSES if vulnerabilities are found
        self.assertGreater(
            pickle_loads,
            0,
            "EXPECTED: Language modeling dataset should contain pickle.load vulnerabilities. "
            "This is a HIGH-RISK area as it loads cached features."
        )
        
        print(f"\n[HIGH-RISK VULNERABILITY] Language modeling dataset contains {pickle_loads} pickle.load calls")

    def test_07_check_marian_conversion_yaml_vulnerability(self):
        """
        Test 7: Check Marian conversion script for unsafe YAML loading
        
        The Marian model conversion script uses yaml.load with BaseLoader, which
        can execute arbitrary code.
        """
        marian_file = self.repo_root / "src" / "transformers" / "models" / "marian" / "convert_marian_to_pytorch.py"
        
        if not marian_file.exists():
            self.skipTest("Marian conversion file not found")
        
        with open(marian_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        unsafe_yaml = len(re.findall(r'yaml\.load\([^)]*BaseLoader', content))
        
        # This test PASSES if vulnerabilities are found
        self.assertGreater(
            unsafe_yaml,
            0,
            "EXPECTED: Marian conversion script should contain unsafe yaml.load. "
            "BaseLoader can execute arbitrary code."
        )
        
        print(f"\n[HIGH-RISK VULNERABILITY] Marian conversion script contains {unsafe_yaml} unsafe yaml.load calls")

    def test_08_verify_no_safe_alternatives_used(self):
        """
        Test 8: Verify that safe alternatives are NOT consistently used
        
        This negative test confirms that the codebase is NOT using safer alternatives
        like safetensors or yaml.safe_load consistently.
        """
        src_dir = self.repo_root / "src" / "transformers"
        
        # Check for yaml.safe_load usage
        safe_yaml_count = 0
        unsafe_yaml_count = 0
        
        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    safe_yaml_count += len(re.findall(r'yaml\.safe_load', content))
                    unsafe_yaml_count += len(re.findall(r'yaml\.load\([^)]*Loader\s*=\s*yaml\.(BaseLoader|FullLoader)', content))
            except Exception:
                continue
        
        # The vulnerability exists if unsafe usage is present
        self.assertGreater(
            unsafe_yaml_count,
            0,
            "EXPECTED: Unsafe yaml.load should be present, confirming the vulnerability."
        )
        
        print(f"\n[VULNERABILITY CONFIRMED] YAML loading analysis:")
        print(f"  - Safe yaml.safe_load usage: {safe_yaml_count}")
        print(f"  - Unsafe yaml.load usage: {unsafe_yaml_count}")
        print(f"  - Vulnerability exists: Safe alternatives NOT consistently used")

    def test_09_comprehensive_vulnerability_summary(self):
        """
        Test 9: Generate comprehensive vulnerability summary
        
        This test aggregates all findings to provide a complete picture of the
        insecure deserialization vulnerability across the codebase.
        """
        src_dir = self.repo_root / "src" / "transformers"
        examples_dir = self.repo_root / "examples"
        
        # Count all pickle.load instances
        total_pickle_loads = 0
        affected_files = set()
        
        for directory in [src_dir, examples_dir]:
            if not directory.exists():
                continue
            for py_file in directory.rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        count = len(re.findall(r'\bpickle\.load\(', content))
                        if count > 0:
                            total_pickle_loads += count
                            affected_files.add(str(py_file.relative_to(self.repo_root)))
                except Exception:
                    continue
        
        # This test PASSES to confirm comprehensive vulnerability presence
        self.assertGreater(
            total_pickle_loads,
            10,  # We expect more than 10 instances based on our earlier grep
            f"EXPECTED: Should find significant number of pickle.load vulnerabilities. "
            f"Found {total_pickle_loads} instances across {len(affected_files)} files."
        )
        
        print(f"\n[COMPREHENSIVE VULNERABILITY SUMMARY]")
        print(f"  - Total pickle.load instances: {total_pickle_loads}")
        print(f"  - Total affected files: {len(affected_files)}")
        print(f"  - Severity: CRITICAL")
        print(f"  - Risk: Arbitrary code execution through malicious serialized objects")

    def test_10_verify_cve_001_exists(self):
        """
        Test 10: Final verification that CVE-001 vulnerability exists
        
        This is the master test that confirms the insecure deserialization
        vulnerability described in CVE-001 is present in the codebase.
        """
        vulnerability_found = False
        reasons = []
        
        # Do a fresh scan for this test
        src_dir = self.repo_root / "src" / "transformers"
        examples_dir = self.repo_root / "examples"
        
        # Count pickle.load instances
        pickle_count = 0
        for directory in [src_dir, examples_dir]:
            if not directory.exists():
                continue
            for py_file in directory.rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        pickle_count += len(re.findall(r'\bpickle\.load\(', content))
                except Exception:
                    continue
        
        if pickle_count > 0:
            vulnerability_found = True
            reasons.append(f"Found {pickle_count} pickle.load instances")
        
        # Count unsafe yaml.load instances
        yaml_count = 0
        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    yaml_count += len(re.findall(r'yaml\.load\([^)]*BaseLoader', content))
            except Exception:
                continue
        
        if yaml_count > 0:
            vulnerability_found = True
            reasons.append(f"Found {yaml_count} unsafe yaml.load instances")
        
        # This test PASSES if the vulnerability exists
        self.assertTrue(
            vulnerability_found,
            "CVE-001 vulnerability should exist in the codebase"
        )
        
        print(f"\n{'='*70}")
        print(f"CVE-001 VULNERABILITY CONFIRMED")
        print(f"{'='*70}")
        for reason in reasons:
            print(f"  ✓ {reason}")
        print(f"\nRECOMMENDATION: Replace pickle.load with safetensors.load and")
        print(f"                yaml.load with yaml.safe_load to remediate this")
        print(f"                CRITICAL security vulnerability.")
        print(f"{'='*70}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
