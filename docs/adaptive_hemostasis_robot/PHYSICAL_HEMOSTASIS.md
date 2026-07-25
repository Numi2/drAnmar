# Physical hemostasis contract

The vessel endpoints are held by two explicit fixture-anchor meshes and current-schema vertex attachments. Temporary control uses two independent deformable-to-pad attachments. The pads move through articulated carriages and therefore transmit solver forces into the vessel rather than rewriting vessel transforms. The task-level force envelope targets 1.8 N per pad, flags a soft limit above 4.0 N, and releases both temporary bonds above the 7.0 N hard limit.

A deployed formed clip is an independent rigid body. Two separate attachment volumes on its legs bond to opposite vessel-wall regions. The temporary pad attachments can then be removed while the clip remains load-bearing. A provisional pullout controller removes both leg bonds only when supplied load exceeds its configured threshold.

The patch has eight independent bond cells. Initial placement creates physical vessel-to-patch attachments; cure progression raises the task-level break-force envelope from 0.8 N to 8.0 N over 30 seconds. The stable lane uses a rigid bond carrier, while a portable triangular patch surface is provided for runtime deformable cooking.
