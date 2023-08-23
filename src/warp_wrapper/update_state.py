import torch
from warp.sim import Model, State
import warp as wp
import sys

sys.path.append("src")
from warp_wrapper.state_iterable import StateIterable

wp.init()


class UpdateState(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        force: torch.Tensor,
        position_now: torch.Tensor,
        velocity_now: torch.Tensor,
        model: Model,
        dt: float,
    ):
        ctx.state_now = model.state(requires_grad=True)
        ctx.state_next = model.state(requires_grad=True)
        ctx.tape = wp.Tape()
        ctx.force = wp.from_torch(force.contiguous(), dtype=wp.vec3)
        with ctx.tape:
            wp.sim.collide(model, ctx.state_now)
            ctx.state_now.clear_forces()
            ctx.state_now.particle_f = ctx.force
            wp.sim.SemiImplicitIntegrator().simulate(model, ctx.state_now, ctx.state_next, dt)
        position_next = wp.to_torch(ctx.state_next.particle_q).requires_grad_()
        velocity_next = wp.to_torch(ctx.state_next.particle_qd).requires_grad_()
        return position_next, velocity_next

    @staticmethod
    def backward(ctx, grad_position_next, grad_velocity_next):
        ctx.state_next.particle_q.grad = wp.from_torch(grad_position_next.contiguous(), dtype=wp.vec3)
        ctx.state_next.particle_qd.grad = wp.from_torch(grad_velocity_next.contiguous(), dtype=wp.vec3)
        ctx.tape.backward()

        return (
            wp.to_torch(ctx.tape.gradients[ctx.force]).requires_grad_(),
            wp.to_torch(ctx.tape.gradients[ctx.state_now.particle_q]).requires_grad_(),
            wp.to_torch(ctx.tape.gradients[ctx.state_now.particle_qd]).requires_grad_(),
            None,
            None,
            None,
            None
        )
