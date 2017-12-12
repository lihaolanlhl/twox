import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { TwoxComponent } from './twox.component';

describe('TwoxComponent', () => {
  let component: TwoxComponent;
  let fixture: ComponentFixture<TwoxComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ TwoxComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(TwoxComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
