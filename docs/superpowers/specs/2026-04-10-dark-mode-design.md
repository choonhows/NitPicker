# Dark Mode Design — NitPicker

**Date:** 2026-04-10

## Overview

Add a user-toggled dark mode to NitPicker that persists across page refreshes via `localStorage`. The implementation uses Tailwind CSS v4's class-based dark mode and a lightweight custom hook — no additional dependencies required.

## Architecture

### Tailwind v4 Dark Mode Config

Add to `src/index.css`:

```css
@variant dark (&:where(.dark, .dark *));
```

This registers the `dark:` variant to activate when any ancestor element has the `dark` class. The `dark` class is placed on `<html>`.

### `src/hooks/useTheme.ts`

A single custom hook — no React context needed. The `dark` class on `<html>` is the source of truth; the hook just manages reading/writing it.

- On mount: reads `localStorage.getItem('theme')`, applies `dark` class to `<html>` if value is `'dark'`
- `toggle()`: flips the `dark` class on `<html>`, writes `'dark'` or `'light'` to `localStorage`
- Returns `{ isDark: boolean, toggle: () => void }`

### Toggle Button

Added to `src/components/navbar.tsx`, right side of the nav alongside the existing links. Uses `lucide-react` (already installed):
- Light mode: shows `Moon` icon
- Dark mode: shows `Sun` icon
- No background, minimal styling to match the navbar aesthetic

## Color Palette

The site is high-contrast monochrome. Dark mode inverts the key values:

| Element | Light | Dark |
|---|---|---|
| Page background | `bg-white` | `dark:bg-zinc-950` |
| Primary text | `text-black` | `dark:text-white` |
| Borders | `border-black` | `dark:border-white` |
| Black filled buttons | `bg-black text-white` | `dark:bg-white dark:text-black` |
| Nav hover background | `hover:bg-black` | `dark:hover:bg-white` |
| Nav link hover text | `group-hover:text-white` | `dark:group-hover:text-black` |

## Files Changed

| File | Change |
|---|---|
| `src/index.css` | Add `@variant dark` config |
| `src/hooks/useTheme.ts` | New file — theme hook |
| `src/components/navbar.tsx` | Toggle button + `dark:` classes |
| `src/components/footer.tsx` | `dark:` classes for border and text |
| `src/pages/homepage.tsx` | `dark:` classes throughout |
| `src/pages/mockexam.tsx` | `dark:` classes throughout |
| `src/pages/mockexamprep.tsx` | `dark:` classes throughout |
| `src/pages/mockexamresults.tsx` | `dark:` classes throughout |
| `src/pages/notes.tsx` | `dark:` classes throughout |
| `src/pages/previousexam.tsx` | `dark:` classes throughout |

## Future Pages

The `dark` class on `<html>` is always active once set. Any new page automatically responds to dark mode as long as its elements use paired `dark:` Tailwind utilities (e.g., `bg-white dark:bg-zinc-950`). The infrastructure is automatic; the per-element styling is the developer's responsibility.

## Non-Goals

- System preference auto-detection (`prefers-color-scheme`) — not implemented; user toggle only
- Animated theme transition — not in scope
