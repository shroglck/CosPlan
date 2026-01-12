"""
Domain-Specific Prompt Templates

This module contains prompt templates for different planning domains:
- BlockWorld: Block stacking problems
- Shuffle: Image patch rearrangement
- RoboVQA: Robot manipulation tasks  
- PathPlanning: Grid-based pathfinding

License: MIT (or your preferred license)
"""

# ============================================================================
# BlockWorld Domain Prompts
# ============================================================================

BLOCKWORLD_PROMPTS = {
    "info": """BlockWorld State Tracking and Move Validation Prompt  
You are tasked with analyzing BlockWorld problems where blocks must be moved between columns to achieve a target configuration. Follow these precise guidelines:

#### State Management  
- Track initial positions of all blocks by column number.  
- Update the state after each prerequisite move to maintain accuracy.  
- Use the updated state as the starting point for evaluating solution options.

#### Move Rules
- Blocks can only be moved to the top of other blocks or empty columns
- A block can only be moved if there are no blocks above it
- Blocks can only be placed on top of other blocks or in a column with no previous blocks
- Each move is described as: Move block [ID] from column [X] to column [Y]
""",

    "scene_graph": """You are an expert system for analyzing planning problems using scene graphs. Your task is to create detailed scene graphs for both initial and goal states.

Examine a composite image showing two stages: the left side depicts the initial stage of a task, and the right side illustrates the completed task.

Analyze the initial and goal states to create structured scene graphs. For each state:

1. OBJECTS: Identify all entities (e.g., blocks, puzzle pieces, maze cells, agents)
   - Assign unique identifiers
   - Note their type/category
   - Record observable properties

2. STATES: For each object, capture:
   - Current configuration/position
   - Internal state if applicable
   - Any constraints or restrictions

3. RELATIONSHIPS: Document:
   - Spatial connections (adjacent, above, contains)
   - Logical relationships (part-of, connected-to)
   - Valid transitions or movements
   - Accessibility constraints

4. ENVIRONMENT: Record:
   - Global constraints
   - Boundary conditions
   - Valid action space
   - System rules

Use consistent notation: {{"name": "entity_id", "type": "category", "state": "current_state", "position": "coordinates/location", "properties": {{"key": "value"}}}}

Output:
{{
  "initial_scene_graph": {{
    "objects": [
      {{"name": "entity_1", "type": "category", "state": "state_desc", "position": "location", "properties": {{}}}},
      // Additional entities...
    ]
  }},
  "target_scene_graph": {{
    "objects": [
      {{"name": "entity_1", "type": "category", "state": "target_state", "position": "target_location", "properties": {{}}}},
      // Additional entities...
    ]
  }}
}}""",

    "state_transition": """Given the current state and action sequence, simulate state transitions and generate intermediate scene graphs.

Examine a composite image showing two stages: the left side depicts the initial stage of a task, and the right side illustrates the completed task.

The following steps have already been performed: {previous_steps}

Given the starting scene graph {start_scene_graph}:

1. STATE TRANSITION:
   - Apply action effects
   - Update affected entities
   - Validate against rules
   - Track state changes

2. CONSTRAINT CHECKING:
   - Verify physical/logical constraints
   - Check boundary conditions
   - Validate action preconditions
   - Ensure rule compliance

3. RELATIONSHIP UPDATES:
   - Modify spatial connections
   - Update logical relationships
   - Track dependency changes
   - Maintain consistency

Document ALL changes in format:
{{"step": N, "action": "operation", "affected_entities": [], "state_changes": [], "validity": boolean}}

Output:
{{
  "intermediate_scene_graph": {{
    "objects": [
      {{"name": "entity_1", "type": "category", "state": "current_state", "position": "current_location", "properties": {{}}}},
      // Additional entities...
    ],
    "action_history": [
      {{"step": 1, "action": "operation", "affected_entities": [], "state_changes": [], "validity": true}}
    ],
    "constraints_satisfied": true
  }}
}}""",

    "option_evaluation": """Examine a composite image showing two stages: the left side depicts the initial stage of a task, and the right side illustrates the completed task.

Assume the intermediate scene graph is given as {intermediate_scene_graph} and the target scene graph is given as {target_scene_graph}.

Simulate the following steps:
{steps_to_simulate}

1. SIMULATION:
   - Execute complete action sequence
   - Track all state transitions
   - Monitor constraint satisfaction
   - Record intermediate states

2. EVALUATION METRICS:
   - Goal state alignment (0-100)
   - Constraint satisfaction (0-100)
   - Path efficiency/optimality
   - Resource utilization

3. VALIDATION:
   - Verify operation legality
   - Check completeness
   - Assess reversibility
   - Identify side effects

Compare the simulated scene graph with the target scene and provide the output in the following format.
Compute the similarity score, a number between 0-100, based on the following criteria:
   - Goal state achievement
   - Place of the objects has to match exactly to the position in target scene graph
   - Placement of objects where they are not present in the target scene graph must be heavily penalized
   - Making unnecessary mistakes should be penalized
   - Each difference between simulated state and goal state must be heavily penalized
   - Invalid moves must be heavily penalized
   - Constraint satisfaction
   - Solution efficiency
   - Resource optimization

Output:
{{
  "option_evaluations": [
    {{
      "scene_graph": {{
        "objects": [
          {{"name": "entity_1", "type": "category", "state": "final_state", "position": "final_location", "properties": {{}}}}
        ]
      }},
      "metrics": {{
        "similarity with target scene graph": "similarity score between 0-100"
      }}
    }}
  ]
}}

No other explanation needed."""
}


# ============================================================================
# Mosaic/Puzzle Domain Prompts
# ============================================================================

SHUFFLE_PROMPTS = {
    "info": """Shuffle Image Rearrangement State Tracking and Swap Validation Prompt  

You are tasked with analyzing Shuffle Rearrangement problems where image patches must be swapped to transform a shuffled image into a target configuration.

#### State Management  
- Track initial positions of all patches by their (row, column) coordinates.  
- Update the state after each prerequisite swap to maintain accuracy.  
- Use the updated state as the starting point for evaluating solution options.
- The left image shows the shuffled starting state and the right image shows the target state to be achieved.
- Both images represent the same scene, just with patches in different arrangements.
""",

    "scene_graph": """You are an expert system for analyzing image mosaic rearrangement problems using scene graphs.

Examine a composite image showing two stages: the left side depicts the shuffled image (initial state), and the right side illustrates the correctly arranged image (goal state).

Analyze the initial and goal states to create structured scene graphs. For each state:

1. PATCHES: Identify all image patches
   - Assign unique identifiers based on (row, column) coordinates
   - Note visual content/features of each patch
   - Record observable properties (colors, textures, edge features)

2. STATES: For each patch, capture:
   - Current position in the grid
   - Visual features that help identify it
   - Connections to adjacent patches

3. RELATIONSHIPS: Document:
   - Spatial connections between patches
   - Visual continuity between adjacent patches
   - Edge matching potential
   - Content coherence

4. ENVIRONMENT: Record:
   - Grid dimensions
   - Boundary conditions
   - Valid swap operations
   - System rules

Use consistent notation: {{"name": "patch_(row,col)", "content": "visual_description", "position": "(row,col)", "properties": {{"key": "value"}}}}

Output:
{{
  "initial_scene_graph": {{
    "patches": [
      {{"name": "patch_(0,0)", "content": "visual_description", "position": "(0,0)", "properties": {{}}}},
      // Additional patches...
    ]
  }},
  "target_scene_graph": {{
    "patches": [
      {{"name": "patch_(0,0)", "content": "visual_description", "position": "(0,0)", "properties": {{}}}},
      // Additional patches...
    ]
  }}
}}""",

    "state_transition": """Given the current state and swap sequence, simulate state transitions and generate intermediate scene graphs.

Examine a composite image showing two stages: the left side depicts the shuffled image (initial state), and the right side illustrates the correctly arranged image (goal state).

Rules:
- Image patches can only be swapped with each other, maintaining the same grid structure
- Each swap is described as: Swap patch at position (row1, col1) with patch at position (row2, col2)
- The grid position remains fixed; only the patch content moves

The following steps have already been performed: {previous_steps}

Given the starting scene graph {start_scene_graph}:

1. STATE TRANSITION:
   - Apply swap effects
   - Update affected patch positions
   - Validate against rules
   - Track state changes

2. CONSTRAINT CHECKING:
   - Verify physical grid constraints
   - Check boundary conditions
   - Validate swap preconditions
   - Ensure rule compliance

3. RELATIONSHIP UPDATES:
   - Modify spatial connections
   - Update visual continuity
   - Track position changes
   - Maintain grid consistency

Document ALL changes in format:
{{"step": N, "action": "swap", "affected_patches": [(row1,col1), (row2,col2)], "state_changes": [], "validity": boolean}}

Output:
{{
  "intermediate_scene_graph": {{
    "patches": [
      {{"name": "patch_(0,0)", "content": "visual_description", "position": "(0,0)", "properties": {{}}}},
      // Additional patches...
    ],
    "action_history": [
      {{"step": 1, "action": "swap", "affected_patches": [(0,0), (1,1)], "state_changes": [], "validity": true}}
    ],
    "constraints_satisfied": true
  }}
}}""",

    "option_evaluation": """Examine a composite image showing two stages: the left side depicts the shuffled image (initial state), and the right side illustrates the correctly arranged image (goal state).

Assume the intermediate scene graph is given as {intermediate_scene_graph} and the target scene graph is given as {target_scene_graph}.

Rules:
- Image patches can only be swapped with each other, maintaining the same grid structure
- Each swap is described as: Swap patch at position (row1, col1) with patch at position (row2, col2)

Simulate the following steps:
{steps_to_simulate}

1. SIMULATION:
   - Execute complete swap sequence
   - Track all state transitions
   - Monitor grid integrity
   - Record intermediate states

2. EVALUATION METRICS:
   - Goal state alignment (0-100)
   - Visual continuity (0-100)
   - Path efficiency/optimality
   - Minimum number of swaps

3. VALIDATION:
   - Verify swap legality
   - Check completeness
   - Assess visual coherence
   - Identify visual discontinuities

Compare the simulated scene graph with the target scene and compute similarity score (0-100) based on:
   - Goal state achievement
   - Patch content must match exactly to position in target scene graph
   - Visual continuity between adjacent patches
   - Unnecessary swaps should be penalized
   - Each difference between simulated and goal state must be heavily penalized
   - Invalid swaps must be heavily penalized
   - Solution efficiency (minimum number of swaps)

Output:
{{
  "option_evaluations": [
    {{
      "scene_graph": {{
        "patches": [
          {{"name": "patch_(0,0)", "content": "visual_description", "position": "(0,0)", "properties": {{}}}}
        ]
      }},
      "metrics": {{
        "similarity with target scene graph": "similarity score between 0-100"
      }}
    }}
  ]
}}

No other explanation needed."""
}


# ============================================================================
# RoboVQA Domain Prompts
# ============================================================================

ROBOVQA_PROMPTS = {
    "scene_graph": """You are an expert system for analyzing planning problems using scene graphs.

Examine a composite image showing two stages: the left side depicts the initial stage of a task, and the right side illustrates the completed task.

Analyze the initial and goal states to create structured scene graphs. For each state:

1. OBJECTS: Identify all entities (e.g., blocks, puzzle pieces, maze cells, agents)
   - Assign unique identifiers
   - Note their type/category
   - Record observable properties

2. STATES: For each object, capture:
   - Current configuration/position
   - Internal state if applicable
   - Any constraints or restrictions

3. RELATIONSHIPS: Document:
   - Spatial connections (e.g., adjacent, above, contains)
   - Logical relationships (e.g., part-of, connected-to)
   - Valid transitions or movements
   - Accessibility constraints

4. ENVIRONMENT: Record:
   - Global constraints
   - Boundary conditions
   - Valid action space
   - System rules

Use consistent notation: {{"name": "entity_id", "type": "category", "state": "current_state", "position": "coordinates/location", "properties": {{"key": "value"}}}}

Output:
{{
  "initial_scene_graph": {{
    "objects": [
      {{"name": "entity_1", "type": "category", "state": "state_desc", "position": "location", "properties": {{}}}},
      // Additional entities...
    ]
  }},
  "target_scene_graph": {{
    "objects": [
      {{"name": "entity_1", "type": "category", "state": "target_state", "position": "target_location", "properties": {{}}}},
      // Additional entities...
    ]
  }}
}}""",

    "state_transition": """Given the current state and action sequence, simulate state transitions and generate intermediate scene graphs.

Examine a composite image showing two stages—the left side depicts the initial stage of a task, and the right side illustrates the completed task.

Here is a sequence of actions that have already been executed: {previous_steps}

Given the starting scene graph {start_scene_graph}:

1. STATE TRANSITION:
   - Apply action effects
   - Update affected entities
   - Validate against rules
   - Track state changes

2. CONSTRAINT CHECKING:
   - Verify physical/logical constraints
   - Check boundary conditions
   - Validate action preconditions
   - Ensure rule compliance

3. RELATIONSHIP UPDATES:
   - Modify spatial connections
   - Update logical relationships
   - Track dependency changes
   - Maintain consistency

Document ALL changes in format:
{{"step": N, "action": "operation", "affected_entities": [], "state_changes": [], "validity": boolean}}

Output:
{{
  "intermediate_scene_graph": {{
    "objects": [
      {{"name": "entity_1", "type": "category", "state": "current_state", "position": "current_location", "properties": {{}}}},
      // Additional entities...
    ],
    "action_history": [
      {{"step": 1, "action": "operation", "affected_entities": [], "state_changes": [], "validity": true}}
    ],
    "constraints_satisfied": true
  }}
}}""",

    "option_evaluation": """Examine a composite image showing two stages—the left side depicts the initial stage of a task, and the right side illustrates the completed task.

Assume the intermediate scene graph is given as {intermediate} and the target scene graph is given as {target}.

Simulate the following actions:
{steps_to_simulate}

1. SIMULATION:
   - Execute complete action sequence
   - Track all state transitions
   - Monitor constraint satisfaction
   - Record intermediate states

2. EVALUATION METRICS:
   - Goal state alignment (0-100)
   - Constraint satisfaction (0-100)
   - Path efficiency/optimality
   - Resource utilization

3. VALIDATION:
   - Verify operation legality
   - Check completeness
   - Assess reversibility
   - Identify side effects

Compare the simulated scene graph with the target scene and compute similarity score (0-100) based on:
   - Goal state achievement
   - Objects must match exact positions
   - Objects present where they shouldn't be must be heavily penalized
   - Constraint satisfaction
   - Solution efficiency
   - Resource optimization

Output:
{{
  "option_evaluations": [
    {{
      "scene_graph": {{
        "objects": [
          {{"name": "entity_1", "type": "category", "state": "final_state", "position": "final_location", "properties": {{}}}}
        ]
      }},
      "metrics": {{
        "similarity with target scene graph": "similarity score between 0-100"
      }}
    }}
  ]
}}"""
}


# ============================================================================
# Path Planning Domain Prompts
# ============================================================================

PATH_PLANNING_PROMPTS = {
    "scene_graph": """You are an expert system for analyzing planning problems using scene graphs.

Examine a composite image showing a checkerboard maze with a green circle (start), blue circle (goal), and red rectangular obstacles.

Analyze the initial and goal states to create structured scene graphs. For each state:

1. OBJECTS: Identify all entities (e.g., grid cells, agent, goal, obstacles)
   - Assign unique identifiers
   - Note their type/category
   - Record observable properties

2. STATES: For each object, capture:
   - Current configuration/position
   - Internal state if applicable
   - Any constraints or restrictions

3. RELATIONSHIPS: Document:
   - Spatial connections (e.g., adjacent, above, contains)
   - Logical relationships (e.g., accessible, blocked)
   - Valid transitions or movements
   - Accessibility constraints

4. ENVIRONMENT: Record:
   - Global constraints (e.g., only horizontal/vertical moves)
   - Boundary conditions (e.g., grid size)
   - Valid action space (e.g., up, down, left, right)
   - System rules (e.g., no diagonal moves, avoid obstacles)

Use consistent notation: {{"name": "entity_id", "type": "category", "state": "current_state", "position": "coordinates/location", "properties": {{"key": "value"}}}}

Output:
{{
  "initial_scene_graph": {{
    "objects": [
      {{"name": "agent", "type": "green_circle", "state": "active", "position": [row, col], "properties": {{}}}},
      {{"name": "goal", "type": "blue_circle", "state": "target", "position": [row, col], "properties": {{}}}},
      {{"name": "obstacle_1", "type": "red_rectangle", "state": "blocking", "position": [row, col], "properties": {{}}}}
      // Additional entities...
    ],
    "grid": {{"dimensions": [rows, cols], "coordinates": "(0,0) is top-left"}}
  }},
  "target_scene_graph": {{
    "objects": [
      {{"name": "agent", "type": "green_circle", "state": "active", "position": [row, col], "properties": {{}}}},
      {{"name": "goal", "type": "blue_circle", "state": "target", "position": [row, col], "properties": {{}}}},
      {{"name": "obstacle_1", "type": "red_rectangle", "state": "blocking", "position": [row, col], "properties": {{}}}}
      // Additional entities...
    ],
    "grid": {{"dimensions": [rows, cols], "coordinates": "(0,0) is top-left"}}
  }}
}}""",

    "state_transition": """Given the current state and action sequence, simulate state transitions and generate intermediate scene graphs.

Examine the checkerboard maze where the green circle has already moved along a path.

Given the starting scene graph {start_scene_graph}:

1. STATE TRANSITION:
   - Apply action effects (move in specified direction)
   - Update affected entities (agent position)
   - Validate against rules (no diagonal moves)
   - Track state changes (path history)

2. CONSTRAINT CHECKING:
   - Verify physical constraints (no obstacle collision)
   - Check boundary conditions (stay within grid)
   - Validate action preconditions (move one cell at a time)
   - Ensure rule compliance (horizontal/vertical only)

3. RELATIONSHIP UPDATES:
   - Modify spatial connections (agent's new position)
   - Update logical relationships (accessible cells)
   - Track dependency changes (path taken so far)
   - Maintain consistency (valid moves from new position)

The following steps have already been performed: {previous_steps}

Document ALL changes in format:
{{"step": N, "action": "move_direction", "from": [row, col], "to": [row, col], "validity": boolean, "reason": "explanation if invalid"}}

Output:
{{
  "intermediate_scene_graph": {{
    "objects": [
      {{"name": "agent", "type": "green_circle", "state": "active", "position": [row, col], "properties": {{}}}},
      {{"name": "goal", "type": "blue_circle", "state": "target", "position": [row, col], "properties": {{}}}},
      {{"name": "obstacle_1", "type": "red_rectangle", "state": "blocking", "position": [row, col], "properties": {{}}}}
      // Additional entities...
    ],
    "action_history": [
      {{"step": 1, "action": "move_direction", "from": [row, col], "to": [row, col], "validity": true}}
    ],
    "constraints_satisfied": true,
    "goal_reached": boolean
  }}
}}""",

    "option_evaluation": """Examine the checkerboard maze pathfinding problem.

Assume the intermediate scene graph is given as {intermediate} and the target scene graph is given as {target}.

Simulate the following paths:
{steps_to_simulate}

1. SIMULATION:
   - Execute complete path sequence
   - Track all state transitions
   - Monitor constraint satisfaction
   - Record intermediate states

2. EVALUATION METRICS:
   - Goal state alignment (0-100)
   - Constraint satisfaction (0-100)
   - Path efficiency/optimality
   - Obstacle avoidance

3. VALIDATION:
   - Verify move legality
   - Check completeness (reaches goal)
   - Assess path validity
   - Identify potential issues

Compare the simulated path with the target scene and compute similarity score (0-100) based on:
   - Goal state achievement (reaches blue circle)
   - Position matching (follows valid path)
   - Obstacle avoidance (doesn't cross red rectangles)
   - Constraint satisfaction (only horizontal/vertical moves)
   - Solution efficiency (shortest possible path)

Output:
{{
  "option_evaluations": [
    {{
      "path": [
        [row, col],
        [row, col],
        // Additional steps...
      ],
      "metrics": {{
        "similarity_score": "score between 0-100",
        "reaches_goal": boolean,
        "avoids_obstacles": boolean,
        "uses_legal_moves": boolean,
        "path_length": number
      }},
      "step_by_step": [
        {{"step": 1, "action": "move_direction", "from": [row, col], "to": [row, col], "valid": boolean}},
        // Additional steps...
      ]
    }}
  ],
  "recommended_path": {{
    "coordinate_list": [[row, col], [row, col]],
    "directions": "Step-by-step directions from start to goal"
  }}
}}"""
}


# ============================================================================
# Prompt Registry
# ============================================================================

DOMAIN_PROMPTS = {
    "blockworld": BLOCKWORLD_PROMPTS,
    "shuffle": SHUFFLE_PROMPTS,
    "robovqa": ROBOVQA_PROMPTS,
    "path": PATH_PLANNING_PROMPTS,
}


def get_prompts(domain: str) -> dict:
    """
    Get prompts for a specific domain.
    
    Args:
        domain: Domain name (blockworld, shuffle, robovqa, path)
        
    Returns:
        Dictionary of prompts for the domain
        
    Raises:
        ValueError: If domain is not recognized
    """
    domain_lower = domain.lower()
    if domain_lower not in DOMAIN_PROMPTS:
        raise ValueError(
            f"Unknown domain: {domain}. "
            f"Available domains: {list(DOMAIN_PROMPTS.keys())}"
        )
    
    return DOMAIN_PROMPTS[domain_lower]