import torch
from collections import deque

from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.managers import ManagerTermBase

class AutomaticDomainRandomizationCfg(EventTerm):
    def __init__(self,
                 window_size: int = 1000,
                 target_success_rate: float = 0.9,
                 difficulty_step: float = 0.02,
                 frequency: int = 1000,
                 update_threshold: float = 0.05,
                 **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.target_success_rate = target_success_rate
        self.difficulty_step = difficulty_step
        self.frequency = frequency
        self.update_threshold = update_threshold

class automatic_domain_randomization(ManagerTermBase):
    """Unified controller that tracks success rate and adjusts difficulty automatically.
    
    This controller combines success rate tracking with adaptive difficulty adjustment,
    maintaining a difficulty level from 0 to 1 that increases when the target
    success rate is met and decreases when it falls below target.
    """
    
    def __init__(self, 
                 cfg: AutomaticDomainRandomizationCfg,
                 env: ManagerBasedEnv,
    ):
        """Initialize the adaptive curriculum controller.
        
        Args:
            window_size: Number of episodes to track for success rate calculation
            target_success_rate: Target success rate for curriculum adjustment
            difficulty_step: Amount to change difficulty level each adjustment (default: 0.02)
            frequency: Minimum episodes between adjustments
        """
        # Success rate tracking
        self.window_size = cfg.window_size
        self.target_success_rate = cfg.target_success_rate
        self.success_history = deque(maxlen=cfg.window_size)
        self.episode_count = 0
        
        # Difficulty tracking
        self.difficulty_level = 0.0  # Start at easiest level
        self.difficulty_step = cfg.difficulty_step
        self.max_difficulty = 1.0
        self.min_difficulty = 0.0
        
        # Track when we last adjusted difficulty to avoid too frequent changes
        self.last_adjustment_episode = 0
        self.frequency = cfg.frequency
        
        # Store latest successes for processing during compute
        self._pending_successes = []
        self.update_threshold = cfg.update_threshold
    
    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor):
        """Update the difficulty level based on the success rate."""
        successes = env.unwrapped.reward_manager._episode_sums["success"][env_ids] > 0
        self.compute_and_update(successes)
        self.set_difficulty_level(env)
        
    
    def compute_and_update(self, successes) -> dict[str, float]:
        """Process pending successes and update difficulty if needed.
        
        Args:
           
            
        Returns:
            Dictionary of current state metrics
        """
        # Process any pending successes
        for success in successes:
            self.success_history.append(float(success))
        self.episode_count += len(successes)
        
        # Update difficulty based on current success rate
        difficulty_changed = self._update_difficulty()
        
    
    def _update_difficulty(self) -> bool:
        """Update difficulty level based on current success rate.
        
        Args:
            current_episode: Current episode number (for tracking adjustment frequency)
            
        Returns:
            True if difficulty was adjusted, False otherwise
        """
        # Check if we have enough data and enough time has passed since last adjustment
        if self.episode_count - self.last_adjustment_episode < self.frequency:
            return False
        
        # Check if we should adjust curriculum
        should_increase = self._should_increase_difficulty()
        should_decrease = self._should_decrease_difficulty()
        
        if not (should_increase or should_decrease):
            return False
            
        old_difficulty = self.difficulty_level
        
        if should_increase:
            # Increase difficulty if success rate is above target
            self.difficulty_level = min(self.max_difficulty, 
                                      self.difficulty_level + self.difficulty_step)
        elif should_decrease:
            # Decrease difficulty if success rate is below target
            self.difficulty_level = max(self.min_difficulty, 
                                      self.difficulty_level - self.difficulty_step)
        
        self.last_adjustment_episode = self.episode_count
        # Return True if difficulty actually changed
        return abs(self.difficulty_level - old_difficulty) > 1e-6
    
    def get_success_rate(self) -> float:
        """Get current success rate over the rolling window."""
        if len(self.success_history) == 0:
            return 0.0
        return sum(self.success_history) / len(self.success_history)
    
    def _should_increase_difficulty(self) -> bool:
        """Check if difficulty should be increased."""
        if len(self.success_history) < self.window_size // 2:  # Need enough data
            return False
        success_rate = self.get_success_rate()
        return success_rate > (self.target_success_rate + self.update_threshold)
    
    def _should_decrease_difficulty(self, threshold: float = 0.05) -> bool:
        """Check if difficulty should be decreased."""
        if len(self.success_history) < self.window_size // 2:  # Need enough data
            return False
        success_rate = self.get_success_rate()
        return success_rate < (self.target_success_rate - self.update_threshold)
    
    def set_difficulty_level(self, env: ManagerBasedEnv):
        for term_name in ["randomize_bolt_pose"]:
            term = env.unwrapped.event_manager.get_term_cfg(term_name).func
            term.difficulty_level = self.difficulty_level
    