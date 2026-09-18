# Diagnostics from g1_arm_ik_node and g1_arm_control_node.

string side                       # "left" or "right"
string[] joint_names              # solution vector layout, root -> leaf
float64[] q_current               # from /g1/joint_states (or the mock robot)
float64[] q_target                # last requested target configuration
float64[] q_command               # what the control node is actually sending

float64 arm_sdk_weight            # 0..1 handover weight for the arm_sdk
float64 ramp_progress             # 0..1 of the handover ramp
float64 target_error              # max |q_command - q_current|, radians
float64 position_error            # last IK residual, metres

bool ik_ok                        # last solve within tolerance
bool watchdog_tripped             # state stream went stale
string state                      # DISABLED|ENGAGING|ACTIVE|HOLDING|NO_STATE|REJECTED
string backend                    # mock|topic|dds
int32 cmds_sent
int32 cmds_rejected
string last_reject_reason
