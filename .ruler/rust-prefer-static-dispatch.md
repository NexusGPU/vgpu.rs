# Prefer Static Dispatch Over Dynamic Dispatch

Avoid using dynamic dispatch (dyn Trait) unless strictly necessary. Whenever
possible, prefer static dispatch via impl Trait or generic parameters. This
improves performance through compile-time monomorphization and avoids the
overhead of vtables and heap allocations.

Replace Box<dyn Trait>, &dyn Trait, etc., with static alternatives unless
dynamic polymorphism is required.

Use generic parameters with trait bounds for flexibility and zero-cost
abstraction.

Only retain dynamic dispatch where:

The type is not known at compile time.

Heterogeneous collections or trait object safety is needed.

Ensure all changes preserve functionality and pass all unit tests.Avoid using
dynamic dispatch (dyn Trait) unless strictly necessary. Whenever possible,
prefer static dispatch via impl Trait or generic parameters. This improves
performance through compile-time monomorphization and avoids the overhead of
vtables and heap allocations.

Replace Box<dyn Trait>, &dyn Trait, etc., with static alternatives unless
dynamic polymorphism is required.

Use generic parameters with trait bounds for flexibility and zero-cost
abstraction.

Only retain dynamic dispatch where:

The type is not known at compile time.

Heterogeneous collections or trait object safety is needed.

Ensure all changes preserve functionality and pass all unit tests.
