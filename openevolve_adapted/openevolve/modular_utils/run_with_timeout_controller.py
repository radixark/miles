"""
Universal Timeout Controller for OpenEvolve Programs

Streamlined version supporting both Python and external program execution.
Removed redundant functions while keeping essential multi-language support.
"""

import subprocess
import tempfile  
import time
import os
import signal
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from openevolve.modular_utils.error_constants import ErrorCodes


@dataclass
class ExecutionResult:
    """Standardized execution result for any program type"""
    success: bool                           # Whether execution completed successfully
    exit_code: int                          # Process exit code
    stdout: str                            # Standard output
    stderr: str                            # Standard error
    runtime_seconds: float                 # Actual runtime
    timeout_occurred: bool                 # Whether timeout was hit
    memory_usage_mb: float = 0.0          # Memory usage if available
    
    # Additional execution metadata
    command_executed: List[str] = None     # Actual command that was run
    working_directory: str = ""            # Working directory used
    environment_vars: Dict[str, str] = None
    
    def __post_init__(self):
        if self.command_executed is None:
            self.command_executed = []
        if self.environment_vars is None:
            self.environment_vars = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionResult':
        """Create from dictionary"""
        return cls(**data)


class UniversalTimeoutController:
    """
    Universal controller for executing programs with timeout.
    Supports Python programs, compiled executables, and arbitrary commands.
    """
    
    def __init__(self, default_timeout: int = 300,
                 capture_output: bool = True,
                 debug_verbose: bool = False):
        """
        Initialize timeout controller

        Args:
            default_timeout: Default timeout in seconds
            capture_output: Whether to capture stdout/stderr
            debug_verbose: If True, print stdout/stderr on execution failure
        """
        self.default_timeout = default_timeout
        self.capture_output = capture_output
        self.debug_verbose = debug_verbose
    
    def run_with_timeout(self, command: List[str], 
                        timeout_seconds: Optional[int] = None,
                        working_directory: Optional[str] = None,
                        env_vars: Optional[Dict[str, str]] = None) -> ExecutionResult:
        """
        Execute arbitrary command with timeout
        
        Args:
            command: Command and arguments to execute
            timeout_seconds: Timeout in seconds (uses default if None)
            working_directory: Working directory for execution
            env_vars: Additional environment variables
            
        Returns:
            ExecutionResult with execution details
        """
        if timeout_seconds is None:
            timeout_seconds = self.default_timeout
        
        # Prepare environment
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                capture_output=self.capture_output,
                text=True,
                timeout=timeout_seconds,
                cwd=working_directory,
                env=env
            )
            
            runtime = time.time() - start_time

            if self.debug_verbose and result.returncode != 0:
                print(f"[DEBUG] Program failed with exit code {result.returncode}")
                if self.capture_output and result.stdout:
                    print(f"[DEBUG] STDOUT:\n{result.stdout[:500]}")
                if self.capture_output and result.stderr:
                    print(f"[DEBUG] STDERR:\n{result.stderr[:500]}")

            return ExecutionResult(
                success=result.returncode == 0,
                exit_code=result.returncode,
                stdout=result.stdout if self.capture_output else "",
                stderr=result.stderr if self.capture_output else "",
                runtime_seconds=runtime,
                timeout_occurred=False,
                command_executed=command,
                working_directory=working_directory or os.getcwd(),
                environment_vars=env_vars or {}
            )
            
        except subprocess.TimeoutExpired as e:
            runtime = time.time() - start_time
            stdout_str = e.stdout.decode('utf-8') if e.stdout else ""
            stderr_str = e.stderr.decode('utf-8') if e.stderr else ""

            if self.debug_verbose:
                print(f"[DEBUG] Program timed out after {timeout_seconds}s")
                if stdout_str:
                    print(f"[DEBUG] STDOUT:\n{stdout_str[:500]}")
                if stderr_str:
                    print(f"[DEBUG] STDERR:\n{stderr_str[:500]}")

            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout=stdout_str,
                stderr=stderr_str,
                runtime_seconds=runtime,
                timeout_occurred=True,
                command_executed=command,
                working_directory=working_directory or os.getcwd(),
                environment_vars=env_vars or {}
            )
            
        except Exception as e:
            runtime = time.time() - start_time
            print(f"ERROR: Program execution failed: {str(e)}")
            
            return ExecutionResult(
                success=False,
                exit_code=int(ErrorCodes.EXECUTION_FAILED),  # Use standardized error code
                stdout="",
                stderr=f"Execution error: {str(e)}",
                runtime_seconds=runtime,
                timeout_occurred=False,
                command_executed=command,
                working_directory=working_directory or os.getcwd(),
                environment_vars=env_vars or {}
            )
    
    def run_python_program(self, program_path: str,
                          working_directory: Optional[str] = None,
                          timeout_seconds: Optional[int] = None) -> ExecutionResult:
        """
        Execute Python program with timeout

        Args:
            program_path: Path to Python program
            working_directory: Working directory (defaults to program's directory)
            timeout_seconds: Timeout in seconds

        Returns:
            ExecutionResult with execution details
        """
        # Use program's directory as working directory if not specified
        if working_directory is None:
            working_directory = os.path.dirname(os.path.abspath(program_path))

        # Use python3 and preserve all environment variables from parent process
        # Don't override PYTHONPATH or other critical env vars
        env_vars = {}
        command = ["python3", program_path]
        return self.run_with_timeout(command, timeout_seconds, working_directory, env_vars)
    
    def run_in_temp_workspace(self, program_path: str,
                             timeout_seconds: Optional[int] = None) -> Tuple[ExecutionResult, str]:
        """
        Execute program in temporary workspace
        
        Args:
            program_path: Path to program to execute
            timeout_seconds: Timeout in seconds
            
        Returns:
            Tuple of (ExecutionResult, temp_directory_path)
        """
        temp_dir = tempfile.mkdtemp(prefix="openevolve_exec_")
        
        try:
            result = self.run_python_program(program_path, temp_dir, timeout_seconds)
            return result, temp_dir
        except Exception as e:
            print(f"ERROR: Temp workspace execution failed: {str(e)}")
            return ExecutionResult(
                success=False,
                exit_code=int(ErrorCodes.EXECUTION_FAILED),  # Use standardized error code
                stdout="",
                stderr=f"Workspace execution error: {str(e)}",
                runtime_seconds=0.0,
                timeout_occurred=False,
                command_executed=["python", program_path],
                working_directory=temp_dir
            ), temp_dir


# Convenience functions
def create_timeout_controller(timeout: int = 300, 
                            capture_output: bool = True) -> UniversalTimeoutController:
    """Create timeout controller with specified defaults"""
    return UniversalTimeoutController(
        default_timeout=timeout,
        capture_output=capture_output
    )


if __name__ == "__main__":
    # Test the streamlined timeout controller
    print("=== Testing Streamlined Timeout Controller ===")
    
    controller = create_timeout_controller(timeout=10)
    
    # Test Python execution
    print("Testing Python execution...")
    result = controller.run_with_timeout(["python", "-c", "print('Hello from Python!')"])
    print(f"✓ Python result: success={result.success}, output='{result.stdout.strip()}'")
    
    # Test arbitrary command
    print("Testing arbitrary command...")
    result = controller.run_with_timeout(["echo", "Hello from shell!"])
    print(f"✓ Shell result: success={result.success}, output='{result.stdout.strip()}'")
    
    # Test timeout
    print("Testing timeout...")
    result = controller.run_with_timeout(["python", "-c", "import time; time.sleep(15)"], timeout_seconds=2)
    print(f"✓ Timeout result: success={result.success}, timeout_occurred={result.timeout_occurred}")
    
    print("Streamlined timeout controller test completed!")