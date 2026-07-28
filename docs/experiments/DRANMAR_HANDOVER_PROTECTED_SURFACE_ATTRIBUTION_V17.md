# Protected-Surface Attribution v17

v15 is one protected-surface event above its paired scale
non-inferiority bound. The existing evidence identified the terminal but did
not identify the responsible tool, jaw, or maximum physical phase. Further
controller tuning without that information would be blind.

v17 does not change the policy, task, reward, success definition, physics, or
termination. It adds first-episode evidence for:

- every failure reason crossed with the maximum handover phase;
- maximum PhysX-derived non-object force for each of the four jaw sensors;
- the number of protected-surface episodes crossing 2 N per jaw; and
- attribution to robot 1 or robot 2.

Because Isaac Lab automatically resets terminal environments inside
`env.step`, v17 stores the four-force vector at the termination function
before reset. Post-step sensor snapshots are not used for attribution.

The diagnostic is evaluated on the v15 0.08 phase- and contact-qualified
controller at the exact 2,000-environment population. The next control change
must target the measured collision path.

## Result

All five protected-surface terminals crossed 2 N on robot 2, the receiver.
Robot 1 crossed 2 N in one overlapping episode. Four of five events occurred
after lift and before receiver acquisition; the maximum receiver-jaw force was
4.171 N. v18 therefore leaves giver transport unchanged and decelerates only
the last 6 mm of recovered receiver translation.
