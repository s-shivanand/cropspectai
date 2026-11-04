/** Simulate Chrome streams and native widget clicks without hardware or packages. */
import assert from 'node:assert/strict';
import { ReadableStream, WritableStream } from 'node:stream/web';
import mount, { createSerialDevice } from '../browser_serial.js';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
function fixture(options = {}) {
    let source;
    const commands = [];
    const packets = [];
    const port = {
        opened: 0, closed: 0,
        async open(config) {
            this.opened += 1;
            assert.equal(config.baudRate, 115200);
            this.readable = new ReadableStream({ start(controller) { source = controller; } });
            this.writable = new WritableStream({ write(bytes) {
                const command = new TextDecoder().decode(bytes);
                commands.push(command);
                if (command === 'r') options.batch?.(send);
            } });
        },
        async close() { this.closed += 1; },
    };
    const send = (text) => source.enqueue(new TextEncoder().encode(text));
    const serial = { requests: 0, requestPort() {
        this.requests += 1;
        return options.select ? options.select(port) : Promise.resolve(port);
    } };
    const device = createSerialDevice(serial, (state) => packets.push(state), {
        bootMs: 0, readyMs: 0, settleMs: 0, batchMs: 50, liveMs: 1000, ...options,
    });
    return { device, serial, port, send, commands, packets, source: () => source };
}
const frame = (kind, value = 100) => kind + ',' + Array(12).fill(value).join(',') + '\n';
const tests = [];
const test = (name, run) => tests.push({ name, run });

test('permission is requested synchronously from Start; chunked live data and Stop', async () => {
    const f = fixture();
    const started = f.device.start();
    assert.equal(f.serial.requests, 1);
    await started;
    f.send('DATA,100,100,100');
    f.send(',100,100,100,100,100,100,100,100,100\r\n');
    await sleep(1);
    assert.deepEqual(f.device.state.values, Array(12).fill(100));
    assert.equal(f.device.state.live, true);
    await f.device.stop();
    assert.equal(f.device.state.live, false);
    assert.deepEqual(f.commands, ['s', 'l', 's']);
    await f.device.disconnect();
    assert.equal(f.port.closed, 1);
});

test('measurement requests permission directly and accepts exactly ten RAW frames', async () => {
    const f = fixture({ batch(send) {
        send(frame('RAW', 200).repeat(10) + 'RAW_DONE\n');
    } });
    const measured = f.device.measure({ id: 'batch-1', specimen: 'leaf_01' });
    assert.equal(f.serial.requests, 1);
    await measured;
    assert.equal(f.device.state.raw.length, 10);
    assert.equal(f.device.state.raw[0][0], 200);
    assert.equal(f.device.state.request.specimen, 'leaf_01');
    assert.equal(f.device.state.busy, false);
    assert.equal(f.device.state.error, null);
    await f.device.disconnect();
});

test('live to batch uses the open port and ignores stale DATA frames', async () => {
    const f = fixture({ batch(send) {
        send(frame('DATA') + frame('RAW', 300).repeat(10) + 'RAW_DONE\n');
    } });
    await f.device.start();
    await f.device.measure({ id: 'batch-2', specimen: 'leaf_02' });
    assert.equal(f.serial.requests, 1);
    assert.equal(f.device.state.live, false);
    assert.equal(f.device.state.raw[0][0], 300);
    await f.device.disconnect();
});

for (const [name, batch] of [
    ['short', (send) => send(frame('RAW').repeat(9) + 'RAW_DONE\n')],
    ['long', (send) => send(frame('RAW').repeat(11) + 'RAW_DONE\n')],
    ['invalid', (send) => send(frame('RAW', 'NaN') + 'RAW_DONE\n')],
    ['negative', (send) => send(frame('RAW', -1) + 'RAW_DONE\n')],
    ['aborted', (send) => send('Read failed! Measurement aborted.\n')],
    ['timed out', () => {}],
]) {
    test(`${name} batches never publish measurement data`, async () => {
        const f = fixture({ batch });
        await f.device.measure({ id: name, specimen: 'leaf' });
        assert.equal(f.device.state.raw, null);
        assert.ok(f.device.state.error);
        assert.equal(f.device.state.busy, false);
        await f.device.disconnect();
    });
}

test('cancelled permission can be retried', async () => {
    let rejected = true;
    const f = fixture({ select(port) {
        if (rejected) return Promise.reject(Object.assign(new Error('cancelled'), { name: 'NotFoundError' }));
        return Promise.resolve(port);
    } });
    await f.device.start();
    assert.match(f.device.state.error, /No measuring device selected/);
    rejected = false;
    await f.device.start();
    assert.equal(f.device.state.live, true);
    assert.equal(f.device.state.error, null);
    await f.device.disconnect();
});

test('unplugging during a batch produces a failure and releases locks', async () => {
    const f = fixture({ batch() { f.source().error(new Error('USB unplugged')); } });
    await f.device.measure({ id: 'unplug', specimen: 'leaf' });
    await sleep(5);
    assert.equal(f.device.state.raw, null);
    assert.match(f.device.state.error, /USB unplugged/);
    assert.equal(f.port.closed, 1);
});

test('reset while device selection is pending never opens the selected port', async () => {
    let resolve;
    const f = fixture({ select: (port) => new Promise((done) => { resolve = () => done(port); }) });
    const started = f.device.start();
    await f.device.disconnect();
    resolve();
    await started;
    assert.equal(f.port.opened, 0);
    assert.equal(f.device.state.live, false);
});

test('reset while port.open is pending closes it after opening finishes', async () => {
    const f = fixture();
    const open = f.port.open.bind(f.port);
    let resolveOpen;
    f.port.open = async (config) => {
        await new Promise((resolve) => { resolveOpen = resolve; });
        await open(config);
    };
    const started = f.device.start();
    await sleep(1);
    await f.device.disconnect();
    resolveOpen();
    await started;
    assert.equal(f.port.closed, 1);
    assert.equal(f.device.state.live, false);
});

test('stopping during boot cancels live activation and releases the connection', async () => {
    const f = fixture({ bootMs: 20, readyMs: 0 });
    const started = f.device.start();
    await sleep(1);
    await f.device.stop();
    await started;
    assert.equal(f.device.state.live, false);
    assert.ok(!f.commands.includes('l'));
    assert.equal(f.port.closed, 1);
});

test('stalled live streaming stops with an actionable error', async () => {
    const f = fixture({ liveMs: 5 });
    await f.device.start();
    await sleep(15);
    assert.equal(f.device.state.live, false);
    assert.match(f.device.state.error, /No live readings/);
    await f.device.disconnect();
});

test('bridge uses existing buttons, captures permission gesture, and survives rerenders', async () => {
    const listeners = new Map();
    const packets = [];
    const f = fixture({ batch(send) {
        send(frame('RAW').repeat(10) + 'RAW_DONE\n');
    } });
    const old = new Map(['document', 'window', 'navigator', 'isSecureContext', '__cropSpectSerial']
        .map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
    const set = (key, value) => Object.defineProperty(globalThis, key, { configurable: true, value });
    const root = { isConnected: true, closest: () => ({ style: { setProperty() {} } }) };
    set('document', {
        addEventListener: (name, listener) => listeners.set(name, listener),
        removeEventListener: (name, listener) => { if (listeners.get(name) === listener) listeners.delete(name); },
        querySelector: () => ({ value: 'leaf_01' }),
    });
    set('window', { addEventListener() {}, removeEventListener() {} });
    set('navigator', { serial: f.serial });
    set('isSecureContext', true);
    set('__cropSpectSerial', new Map());
    try {
        const component = { key: 'test', data: { bootMs: 0, readyMs: 0, settleMs: 0 },
            parentElement: root, setStateValue: (key, packet) => packets.push(packet) };
        const cleanup = mount(component);
        const entry = globalThis.__cropSpectSerial.get('test');
        let intercepted = false;
        const button = { closest: (selector) => selector === '.st-key-toggle_live' ? button : null };
        listeners.get('click')({ target: { closest: () => button },
            preventDefault() {}, stopImmediatePropagation() { intercepted = true; } });
        assert.equal(f.serial.requests, 1, 'requestPort must be called inside the original click');
        assert.equal(intercepted, true);
        await sleep(10);
        cleanup();
        const cleanupAgain = mount(component);
        assert.equal(globalThis.__cropSpectSerial.get('test'), entry);
        assert.equal(f.serial.requests, 1);
        await entry.device.stop();
        const measureButton = { closest: (selector) => selector === '.st-key-measure_predict' ? measureButton : null };
        const measureClick = () => listeners.get('click')({ target: { closest: () => measureButton },
            preventDefault() {}, stopImmediatePropagation() {} });
        measureClick();
        await sleep(10);
        const firstRequest = entry.device.state.request.id;
        assert.equal(entry.device.state.raw.length, 10);
        measureClick();
        await sleep(5);
        assert.equal(entry.device.state.request.id, firstRequest, 'wait for server save acknowledgment');
        mount({ ...component, data: { ...component.data, ack: firstRequest } });
        measureClick();
        await sleep(10);
        assert.notEqual(entry.device.state.request.id, firstRequest);
        // Blank labels retain the native form validation and do not ask for USB.
        globalThis.document.querySelector = () => ({ value: '  ' });
        const measure = { closest: (selector) => selector === '.st-key-measure_predict' ? measure : null };
        intercepted = false;
        listeners.get('click')({ target: { closest: () => measure },
            preventDefault() {}, stopImmediatePropagation() { intercepted = true; } });
        assert.equal(intercepted, false);
        root.isConnected = false;
        cleanupAgain();
        await sleep(320);
        assert.equal(globalThis.__cropSpectSerial.size, 0);
        assert.equal(f.port.closed, 1);
    } finally {
        for (const [key, descriptor] of old) {
            if (descriptor) Object.defineProperty(globalThis, key, descriptor);
            else delete globalThis[key];
        }
    }
});

export const results = [];
for (const { name, run } of tests) {
    await run();
    results.push(name);
}
