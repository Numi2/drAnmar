# Robot integration

Every DrAnmar robot reports structured intervention events instead of directly modifying unrelated patient subsystems. The event registry applies the mechanical consequence, updates regional perfusion or bleeding, records damage, and produces one serializable patient snapshot.

The robot modules are optional dependencies. The patient runs without them and accepts equivalent external events through the same contract.
