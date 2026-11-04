/** Acquire the existing ESP32 protocol without rendering or replacing any UI. */
export function createSerialDevice(serial, emit, options = {}) {
    const config = { channels: 12, count: 10, baud: 115200, bootMs: 2000,
        readyMs: 5000, settleMs: 300, batchMs: 20000, liveMs: 15000, ...options };
    const state = { supported: true, live: false, busy: false, values: null,
        error: null, request: null, raw: null };
    let port, reader, writer, readTask, connecting, ready, batch, liveTimer;
    let closing = false;
    let generation = 0;
    let disconnecting;
    const publish = () => emit({ ...state });
    const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
    const command = async (text) => {
        if (!writer) throw new Error('The measuring device is disconnected.');
        await writer.write(new TextEncoder().encode(text));
    };
    const clearLive = () => { clearTimeout(liveTimer); state.live = false; };
    const rejectBatch = (error) => {
        if (batch) { clearTimeout(batch.timer); batch.reject(error); batch = null; }
    };
    const armLiveTimeout = () => {
        clearTimeout(liveTimer);
        liveTimer = setTimeout(() => {
            void command('s').catch(() => {});
            clearLive();
            state.error = 'No live readings received. Check the device and try again.';
            publish();
        }, config.liveMs);
    };
    const parseLine = (line) => {
        if (line === 'AS7343_READY') ready?.();
        if (line === 'RAW_DONE' && batch) {
            const pending = batch;
            batch = null;
            clearTimeout(pending.timer);
            if (pending.rows.length !== config.count) {
                pending.reject(new Error(`Expected ${config.count} complete readings.`));
            } else pending.resolve(pending.rows);
            return;
        }
        if (/failed|aborted/i.test(line) && batch) {
            rejectBatch(new Error('The sensor could not complete the measurement.'));
            return;
        }
        const [kind, ...fields] = line.split(',');
        if (kind !== 'DATA' && kind !== 'RAW') return;
        const values = fields.map(Number);
        if (fields.length !== config.channels || fields.some((v) => !v.trim()) ||
            values.some((v) => !Number.isFinite(v) || v < 0 || v > 65535)) {
            if (kind === 'RAW') rejectBatch(new Error('The sensor returned an invalid raw frame.'));
            return;
        }
        if (kind === 'RAW' && batch) {
            batch.rows.push(values);
            if (batch.rows.length > config.count) {
                rejectBatch(new Error('The sensor returned too many raw readings.'));
            }
        }
        if (kind === 'DATA' && state.live && !state.busy) {
            state.values = values;
            armLiveTimeout();
            publish();
        }
    };
    const read = async () => {
        const decoder = new TextDecoder();
        let buffer = '';
        try {
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                if (buffer.length > 65536) throw new Error('Invalid serial data received.');
                let newline;
                while ((newline = buffer.indexOf('\n')) >= 0) {
                    parseLine(buffer.slice(0, newline).trim());
                    buffer = buffer.slice(newline + 1);
                }
            }
            if (!closing) throw new Error('The measuring device was disconnected.');
        } catch (error) {
            if (!closing) {
                clearLive();
                rejectBatch(error);
                state.error = error.message;
                publish();
                // Let this reader return before disconnect waits for it.
                setTimeout(() => { void disconnect(); }, 0);
            }
        } finally {
            reader?.releaseLock();
            reader = null;
        }
    };
    const connect = () => {
        if (connecting) return connecting;
        if (writer) return Promise.resolve();
        // This call must precede every await: it runs inside the native button click.
        let selection;
        try { selection = serial.requestPort(); }
        catch (error) { return Promise.reject(error); }
        const connectionGeneration = generation;
        connecting = (async () => {
            try {
                const selected = await selection;
                if (connectionGeneration !== generation) return;
                await selected.open({ baudRate: config.baud });
                if (connectionGeneration !== generation) {
                    await selected.close();
                    return;
                }
                port = selected;
                writer = port.writable.getWriter();
                reader = port.readable.getReader();
                let readyTimer;
                const readiness = new Promise((resolve) => {
                    ready = resolve;
                    readyTimer = setTimeout(resolve, config.bootMs + config.readyMs);
                });
                readTask = read();
                await delay(config.bootMs);
                await readiness;
                clearTimeout(readyTimer);
                ready = null;
                if (connectionGeneration === generation) {
                    await command('s');
                    await delay(config.settleMs);
                }
            } catch (error) {
                await disconnect();
                throw error;
            } finally { connecting = null; }
        })();
        return connecting;
    };
    const fail = (error) => {
        clearLive();
        state.busy = false;
        state.error = error.name === 'NotFoundError'
            ? 'No measuring device selected. Press Start or Measure and predict to try again.'
            : `Could not read the measuring device: ${error.message}`;
        publish();
    };
    const start = async () => {
        if (state.busy) return;
        state.busy = true;
        state.live = true;
        state.error = null;
        state.request = null;
        state.raw = null;
        state.values = null;
        publish();
        const operationGeneration = generation;
        try {
            await connect();
            if (operationGeneration !== generation) return;
            state.busy = false;
            state.live = true;
            await command('l');
            armLiveTimeout();
            publish();
        } catch (error) { await disconnect(); fail(error); }
    };
    const stop = async () => {
        if (state.busy && state.request) return;
        if (state.busy) {
            clearLive();
            publish();
            await disconnect();
            state.busy = true;
            await connecting;
            state.busy = false;
            publish();
            return;
        }
        clearLive();
        try { if (writer) await command('s'); }
        catch (error) { fail(error); }
        publish();
    };
    const measure = async (request) => {
        if (state.busy) return;
        state.busy = true;
        state.error = null;
        state.request = request;
        state.raw = null;
        clearLive();
        publish();
        const operationGeneration = generation;
        try {
            await connect();
            if (operationGeneration !== generation) return;
            await command('s');
            // Allow the last live frame to drain before asking for a batch.
            await delay(config.settleMs);
            const completed = new Promise((resolve, reject) => {
                batch = { rows: [], resolve, reject,
                    timer: setTimeout(() => rejectBatch(new Error('Measurement timed out.')),
                        config.batchMs) };
            });
            // Attach rejection before writing, including when a device disconnects mid-write.
            const acquisition = (async () => {
                try { await command('r'); } catch (error) { rejectBatch(error); }
            })();
            state.raw = await completed;
            await acquisition;
            state.busy = false;
            publish();
        } catch (error) { await disconnect(); fail(error); }
    };
    function disconnect() {
        if (disconnecting) return disconnecting;
        generation += 1;
        closing = true;
        clearLive();
        rejectBatch(new Error('Measurement cancelled.'));
        ready?.();
        disconnecting = (async () => {
            try { if (writer) await command('s'); } catch { /* Unplugged port. */ }
            try { await reader?.cancel(); } catch { /* Already disconnected. */ }
            await readTask;
            try { writer?.releaseLock(); } catch { /* Already released. */ }
            writer = null;
            try { await port?.close(); } catch { /* Already closed. */ }
            port = null;
            state.busy = false;
            closing = false;
        })().finally(() => { disconnecting = null; });
        return disconnecting;
    }
    return { start, stop, measure, disconnect, state };
}

/** Attach one invisible bridge to the existing Streamlit widget DOM. */
export default function mount(component) {
    const { key, data, parentElement, setStateValue } = component;
    const registry = globalThis.__cropSpectSerial ??= new Map();
    let entry = registry.get(key);
    if (!entry) {
        entry = { root: parentElement, emit: (packet) => setStateValue('packet', packet) };
        registry.set(key, entry);
        if (!navigator.serial || !globalThis.isSecureContext) {
            entry.emit({ supported: false });
        } else {
            entry.device = createSerialDevice(navigator.serial, (packet) => entry.emit(packet), data);
            const click = (event) => {
                const target = event.target.closest?.('button');
                if (!target || !entry.root.isConnected) return;
                if (target.closest('.st-key-reset_app, .st-key-device_yes, .st-key-device_no')) {
                    // Reset/mode switches still use their original Python callbacks.
                    void entry.device.disconnect();
                    return;
                }
                const toggle = target.closest('.st-key-toggle_live');
                const measure = target.closest('.st-key-measure_predict');
                if (!toggle && !measure) return;
                const specimen = measure
                    ? document.querySelector('.st-key-specimen_label input')?.value.trim() : null;
                if (measure && !specimen) return;
                event.preventDefault();
                event.stopImmediatePropagation();
                if (toggle && entry.device.state.live && !entry.device.state.request) {
                    void entry.device.stop();
                    return;
                }
                if (entry.device.state.busy || (
                    entry.device.state.raw && entry.device.state.request?.id !== entry.ack
                )) return;
                if (toggle) {
                    void (entry.device.state.live ? entry.device.stop() : entry.device.start());
                } else {
                    void entry.device.measure({ id: crypto.randomUUID(), specimen });
                }
            };
            entry.click = click;
            document.addEventListener('click', click, true);
            entry.unload = () => { void entry.device.disconnect(); };
            window.addEventListener('pagehide', entry.unload);
            entry.emit({ ...entry.device.state });
        }
    }
    entry.ack = data.ack;
    entry.root = parentElement;
    entry.emit = (packet) => setStateValue('packet', packet);
    // No HTML or widgets: hide the empty wrapper so it adds no layout space.
    parentElement.closest('[data-testid="stElementContainer"]')?.style.setProperty('display', 'none');
    clearTimeout(entry.cleanup);
    return () => {
        // Streamlit also cleans up on data rerenders. Close only after actual removal.
        entry.cleanup = setTimeout(() => {
            if (entry.root.isConnected) return;
            document.removeEventListener('click', entry.click, true);
            window.removeEventListener('pagehide', entry.unload);
            void entry.device?.disconnect();
            registry.delete(key);
        }, 300);
    };
}
