"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "cn";

const SURFACES = [
  { href: "/live", label: "Live Traffic" },
  { href: "/inspector", label: "Decision Inspector" },
  { href: "/red-team", label: "Red-Team Console" },
] as const;

/**
 * Dense, instrument-panel nav bar - the spec's visual direction is
 * "terminal-adjacent, not neon-cyberpunk," so this stays a plain
 * monospace-labelled strip rather than a decorated sidebar.
 */
export function NavShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b border-border bg-background">
        <div className="flex h-12 items-center gap-1 px-4">
          <span className="mr-4 font-mono text-sm font-semibold tracking-tight">
            PORTCULLIS
          </span>
          <nav className="flex gap-1">
            {SURFACES.map((surface) => (
              <Link
                key={surface.href}
                href={surface.href}
                className={cn(
                  "rounded px-3 py-1.5 font-mono text-xs uppercase tracking-wide transition-colors",
                  pathname === surface.href
                    ? "bg-muted text-foreground"
                    : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
                )}
              >
                {surface.label}
              </Link>
            ))}
          </nav>
        </div>
      </header>
      <main className="flex-1">{children}</main>
    </div>
  );
}
