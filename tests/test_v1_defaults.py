import driver_v1


def test_default_joint_batch_matches_256_agents():
    args = driver_v1.build_parser().parse_args([])
    assert args.joint_batch_size == 64
    assert args.warmup_agent_samples == 2000
    assert args.checkpoint_every == 100
