# Rust Ownership and Borrowing Guidance

When writing Rust code, be deliberate about ownership. Avoid taking or storing
ownership of a value unless it is strictly necessary (e.g., for moving into
long-lived structures, thread transfer, or when mutation/lifetime requirements
demand it).

Prefer borrowing (&T or &mut T) over ownership (T) for function parameters,
return values, and internal data structures whenever possible. This minimizes
unnecessary memory allocations, data copying, and enables better reuse of
resources.

Always consider the cost of cloning or moving large or complex types. If a value
is cheap to borrow but expensive to move or clone, avoid unnecessary ownership
transfer. For example, when passing strings, prefer &str or &String over taking
a full String unless mutation or ownership transfer is truly required.

Furthermore, when designing APIs or data structures, be mindful of lifetime
annotations to ensure borrowed references remain valid without overly
constraining flexibility. In performance-critical code, overuse of owned values
can lead to subtle inefficiencies that compound across the system.When writing
Rust code, be deliberate about ownership. Avoid taking or storing ownership of a
value unless it is strictly necessary (e.g., for moving into long-lived
structures, thread transfer, or when mutation/lifetime requirements demand it).

Prefer borrowing (&T or &mut T) over ownership (T) for function parameters,
return values, and internal data structures whenever possible. This minimizes
unnecessary memory allocations, data copying, and enables better reuse of
resources.

Always consider the cost of cloning or moving large or complex types. If a value
is cheap to borrow but expensive to move or clone, avoid unnecessary ownership
transfer. For example, when passing strings, prefer &str or &String over taking
a full String unless mutation or ownership transfer is truly required.

Furthermore, when designing APIs or data structures, be mindful of lifetime
annotations to ensure borrowed references remain valid without overly
constraining flexibility. In performance-critical code, overuse of owned values
can lead to subtle inefficiencies that compound across the system.
