Prompts = {"basic":"""Examine a composite image showing two stages: the left side depicts the initial stage of a task, and the right side illustrates the completed task.{prompt}""","graph":"""
Examine a composite image showing two stages: the top side depicts the initial stage of a task, and the bottom side illustrates the completed task: To solve the task you are given certain steps
You are a specialized system for analyzing and solving scene state transitions. Your task is to transform an initial state into a desired goal state through a series of precise actions. Follow these steps systematically:
Step 1: Initial Scene Analysis
Begin by creating a detailed scene graph of the initial state:

List all objects present
Describe each object's attributes (position, orientation, state)
Define spatial relationships between objects using precise terms
Format your response as:

Objects: [List each object]
Attributes: [For each object: list key properties]
Relationships: [Define connections between objects]
Constraints: [List any physical or logical constraints]
Step 2: Goal State Definition
Analyze the target state requirements:

Create a parallel scene graph for the goal state
Highlight key differences from initial state
Identify required transformations
Format as:

Target Objects: [List expected objects]
Required Attributes: [Define desired properties]
Goal Relationships: [Specify needed connections]
Success Conditions: [List verification criteria]
Step 3: Action Planning
Develop a strategic action sequence:

Break down the transition into atomic operations
Consider object dependencies and physics constraints
Optimize for minimal steps while maintaining safety
Format as:

Action Sequence:

[Action]: [Expected outcome]

Affected objects: [List]
State changes: [Describe]
Preconditions: [List]


[Continue for each step]

Step 4: Execution and Verification
For each action in your sequence:

Simulate the action's effect on the scene graph
Verify no constraints are violated
Update object states and relationships
Format as:

Action Execution:

Current action: [Describe]
State changes: [List updates]
Verification: [Check constraints]
Updated scene graph: [Show changes]

Step 5: Error Detection
After each action, analyze for:

Unmet goal conditions
Invalid object states
Broken relationships
Format as:

Error Analysis:

Discrepancies: [List differences from goal]
Invalid States: [Describe issues]
Required Corrections: [List needed fixes]

Step 6: Correction Generation
If errors are detected:

Generate minimal corrective actions
Validate corrections maintain existing valid states
Update action sequence
Format as:

Corrections:



Expected resolution: [Outcome]
Validation check: [Verification]



Step 7: Final Validation
Confirm the final state matches all goals:

Compare final scene graph to goal state
Verify all success conditions
Format as:

Final Validation:

Goal Comparison: [List matches/differences]
Success Criteria: [Check each condition]
Final Status: [Complete/Incomplete]

Important Guidelines:

Always think step-by-step
Maintain explicit scene graphs throughout
Verify each action's effects before proceeding
Document any assumptions made
Prioritize safety and stability in transitions
Use precise, unambiguous language
Show your reasoning at each step

For each task provided, follow this structured approach and provide your analysis and solution in the formats specified above. If at any point you identify an impossible transition or conflicting requirements, explain why and suggest alternatives.
Remember: Your goal is to achieve the target state efficiently while maintaining system stability and documenting your process clearly.
Taking into account the above instructions answer the following quesiton {prompt}""","nraph":"""Examine a composite image showing two stages: the top side depicts the initial stage of a task, and the bottom side illustrates the completed task: To solve the task you are given certain steps
You are a specialized system for analyzing and solving scene state transitions. Your task is to transform an initial state into a desired goal state through a series of precise actions. Follow these steps systematically:
Step 1: Initial Scene Analysis
Begin by creating a detailed scene graph of the initial state:

List all objects present
Describe each object's attributes (position, orientation, state)
Define spatial relationships between objects using precise terms
Format your response as:

Objects: [List each object]
Attributes: [For each object: list key properties]
Relationships: [Define connections between objects]
Constraints: [List any physical or logical constraints]
Step 2: Goal State Definition
Analyze the target state requirements:

Create a parallel scene graph for the goal state
Highlight key differences from initial state
Identify required transformations
Format as:

Target Objects: [List expected objects]
Required Attributes: [Define desired properties]
Goal Relationships: [Specify needed connections]
Success Conditions: [List verification criteria]
Examine a composite image showing two stages: the top side depicts the initial stage of a task, and the bottom side illustrates the completed task.{prompt}""","wo_graph":"""Examine the provided image showing two stages: the top shows the initial state, and the bottom shows the desired outcome. Your task is to determine how to transform the initial state into the goal state through logical steps.

Follow this systematic approach:


Step 1: Action Planning
- Create a sequence of steps to achieve the transformation
- Consider dependencies between elements
- Aim for efficiency while maintaining safety
- For each step, describe:
  * What action to take
  * What elements are affected
  * How the state changes
  * What conditions must be met first

Step 2: Step-by-Step Execution
- For each action in your sequence:
  * Describe how it changes the state
  * Verify no constraints are violated
  * Update the current state description

Step 3: Progress Monitoring
- After each action, check for:
  * Progress toward the goal
  * Any unexpected issues

Important Guidelines:
- Think step-by-step
- Track the state of all elements throughout
- Verify each action before proceeding
- Document any assumptions
- Prioritize safety and stability
- Use clear language
- Explain your reasoning

For the given task, follow this structured approach. If you identify any impossible transitions or conflicting requirements, explain why and suggest alternatives.

Remember: Your goal is to achieve the target state efficiently while maintaining stability and clearly explaining your process.{prompt}"""}