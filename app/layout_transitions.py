"""Keep workflow height changes gentle across Streamlit reruns."""

import streamlit as st


def smooth_layout():
    st.iframe(
        """
        <script>
        const win = window.parent;
        const doc = win.document;
        if (!win.cropSpectLayout) {
            const state = {panel: null, content: null, height: 0, animation: null, timer: null};
            win.cropSpectLayout = state;
            const reducedMotion = win.matchMedia('(prefers-reduced-motion: reduce)');
            function resize() {
                const panel = state.panel;
                const content = state.content;
                if (!panel?.isConnected || !content?.isConnected) return;
                const target = content.getBoundingClientRect().height;
                if (Math.abs(target - state.height) < 1) {
                    if (!state.animation || state.animation.playState === 'finished') {
                        panel.style.overflow = '';
                    }
                    return;
                }
                const previous = panel.getBoundingClientRect().height;
                state.animation?.cancel();
                state.height = target;
                panel.style.height = `${target}px`;
                if (reducedMotion.matches) {
                    panel.style.overflow = '';
                    return;
                }
                panel.style.overflow = 'hidden';
                state.animation = panel.animate(
                    [{height: `${previous}px`}, {height: `${target}px`}],
                    {duration: 300, easing: 'cubic-bezier(0.22, 1, 0.36, 1)'}
                );
                state.animation.onfinish = () => { panel.style.overflow = ''; };
            }
            function scheduleResize() {
                if (state.panel) state.panel.style.overflow = 'hidden';
                win.clearTimeout(state.timer);
                state.timer = win.setTimeout(resize, 60);
            }
            const observer = new win.ResizeObserver(scheduleResize);
            function connect() {
                const panel = doc.querySelector('.st-key-workflow_panel');
                const content = doc.querySelector('.st-key-workflow_content');
                if (panel && content && (panel !== state.panel || content !== state.content)) {
                    observer.disconnect();
                    state.animation?.cancel();
                    state.panel = panel;
                    state.content = content;
                    panel.style.height = `${state.height}px`;
                    panel.style.overflow = 'hidden';
                    observer.observe(content);
                    scheduleResize();
                }
            }
            // Streamlit can replace containers during a rerun; rebind only when needed.
            new win.MutationObserver(connect).observe(doc.body, {childList: true, subtree: true});
            reducedMotion.addEventListener('change', () => {
                state.animation?.cancel();
                if (state.panel) state.panel.style.overflow = '';
                scheduleResize();
            });
            connect();
        }
        </script>
        """,
        height=1, tab_index=-1, alt="Smooth workflow transitions",
    )
