use crate::tests::client::MouseEvents;
use crate::tests::fixture::Fixture;
use smithay::utils::Point;
use test_log::test;
use wayland_client::protocol::wl_pointer;
use wayland_protocols::wp::relative_pointer::zv1::client::zwp_relative_pointer_v1;

fn clean_events(events: &mut Vec<MouseEvents>) {
    while let Some(_event) = events.pop() {}
}

#[test]
fn move_mouse() {
    let mut f = Fixture::new();
    f.round_trip();
    f.create_window(320, 240);

    let expected_location = Point::from((0.0, 0.0));
    f.server.pointer_motion_absolute(0, expected_location);
    f.round_trip();

    {
        // Server logic test
        assert_eq!(f.server.pointer_location, expected_location);

        // Client logic test
        let client_events = f.client.get_client_events();
        assert!(client_events.len() >= 1);
        let MouseEvents::Pointer(client_event) = client_events.remove(0) else {
            panic!("Unexpected event: {:?}", client_events);
        };
        let wl_pointer::Event::Enter {
            // First time, we are entering the window
            surface_x,
            surface_y,
            ..
        } = client_event
        else {
            panic!("Unexpected event: {:?}", client_event);
        };
        assert_eq!(surface_x, expected_location.x);
        assert_eq!(surface_y, expected_location.y);

        clean_events(client_events);
    }

    let delta = Point::from((10.0, 15.0));
    f.server.pointer_motion(0, 0, delta, delta);
    f.round_trip();

    {
        // Server logic test
        assert_eq!(f.server.pointer_location, expected_location + delta);

        // Client logic test
        let client_events = f.client.get_client_events();
        assert!(client_events.len() >= 1);
        let MouseEvents::Pointer(client_event) = client_events.remove(0) else {
            panic!("Unexpected event: {:?}", client_events);
        };
        let wl_pointer::Event::Motion {
            // Second time, we are moving thru it
            surface_x,
            surface_y,
            ..
        } = client_event
        else {
            panic!("Unexpected event: {:?}", client_event);
        };
        assert_eq!(surface_x, delta.x);
        assert_eq!(surface_y, delta.y);

        clean_events(client_events);
    }
}

#[test]
fn lock_mouse() {
    let mut f = Fixture::new();
    f.round_trip();
    f.create_window(320, 240);

    let expected_location = Point::from((15.0, 45.0));
    f.server.pointer_motion_absolute(0, expected_location);
    f.round_trip();
    {
        let client_events = f.client.get_client_events();
        assert!(client_events.len() >= 1);
        clean_events(client_events);
    }

    let _lock = f.client.lock_pointer(0, 0, 320, 240);
    let _relative_pointer = f.client.get_relative_pointer();
    f.round_trip();

    // Test pointer_motion()
    let delta = Point::from((10.0, 15.0));
    f.server.pointer_motion(0, 0, delta, delta);
    f.round_trip();
    {
        // Mouse shouldn't be moved!
        assert_eq!(f.server.pointer_location, expected_location);

        // But we should still get Relative mouse events
        let client_events = f.client.get_client_events();
        assert!(client_events.len() >= 2);
        let MouseEvents::Relative(client_event) = client_events.remove(0) else {
            panic!("Unexpected event: {:?}", client_events);
        };
        let zwp_relative_pointer_v1::Event::RelativeMotion { dx, dy, .. } = client_event else {
            panic!("Unexpected event: {:?}", client_event);
        };
        assert_eq!(dx, delta.x);
        assert_eq!(dy, delta.y);

        // And no Pointer Motion events
        while let Some(event) = client_events.pop() {
            match event {
                MouseEvents::Pointer(p_event) => match p_event {
                    wl_pointer::Event::Motion { .. } => panic!("Unexpected event: {:?}", p_event),
                    _ => {}
                },
                _ => {}
            }
        }
    }

    // Test pointer_motion_absolute()
    let absolute_delta = Point::from((100.0, 150.0));
    f.server.pointer_motion_absolute(0, absolute_delta);
    f.round_trip();

    {
        // Mouse shouldn't be moved!
        assert_eq!(f.server.pointer_location, expected_location);

        // But we should still get Relative mouse events
        let client_events = f.client.get_client_events();
        assert!(client_events.len() >= 2);
        let MouseEvents::Relative(client_event) = client_events.remove(0) else {
            panic!("Unexpected event: {:?}", client_events);
        };
        let zwp_relative_pointer_v1::Event::RelativeMotion { dx, dy, .. } = client_event else {
            panic!("Unexpected event: {:?}", client_event);
        };
        assert_eq!(dx, absolute_delta.x);
        assert_eq!(dy, absolute_delta.y);

        // And no Pointer Motion events
        while let Some(event) = client_events.pop() {
            match event {
                MouseEvents::Pointer(p_event) => match p_event {
                    wl_pointer::Event::Motion { .. } => panic!("Unexpected event: {:?}", p_event),
                    _ => {}
                },
                _ => {}
            }
        }
    }
}
