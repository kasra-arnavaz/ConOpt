import torch
from warp.sim import State, Model
import warp as wp

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
        state_now: State,
        state_next: State,
    ):
        ctx.tape = wp.Tape()
        ctx.force = wp.from_torch(force, dtype=wp.vec3)
        with ctx.tape:
            wp.sim.collide(model, state_now)
            state_now.clear_forces()
            state_now.particle_f = ctx.force
            wp.sim.SemiImplicitIntegrator().simulate(model, state_now, state_next, dt)
        ctx.state_next = state_next
        ctx.state_now = state_now
        model.particle_q = state_next.particle_q
        model.particle_qd = state_next.particle_qd
        position_next = wp.to_torch(state_next.particle_q).requires_grad_()
        velocity_next = wp.to_torch(state_next.particle_qd).requires_grad_()
        return position_next, velocity_next

    @staticmethod
    def backward(ctx, grad_position_next, grad_velocity_next):
        ctx.state_next.particle_q.grad = wp.from_torch(grad_position_next, dtype=wp.vec3)
        ctx.state_next.particle_qd.grad = wp.from_torch(grad_velocity_next, dtype=wp.vec3)
        ctx.tape.backward()
        return (
            wp.to_torch(ctx.tape.gradients[ctx.force]).requires_grad_(),
            wp.to_torch(ctx.tape.gradients[ctx.state_now.particle_q]).requires_grad_(),
            wp.to_torch(ctx.tape.gradients[ctx.state_now.particle_qd]).requires_grad_(),
            None,
            None,
            None,
            None,
        )
